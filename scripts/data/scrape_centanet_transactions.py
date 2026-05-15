#!/usr/bin/env python3
"""
Download all Centanet Hong Kong transaction records available from the public
transaction search endpoint and export them to CSV.

Default scope:
  - all Hong Kong
  - both sale/rent transaction types exposed by the site
  - last 1095 days, which is the site's 3-year option
  - all fields returned by the API, flattened with dotted column names

The scraper is resumable. It writes intermediate files next to the CSV:
  - <output>.jsonl       flattened rows, one JSON object per line
  - <output>.state.json  next offset and total count
  - <output>.fields.json discovered CSV columns
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import random
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


API_URL = "https://hk.centanet.com/findproperty/api/Transaction/Search"
DEFAULT_OUTPUT = "data/raw/centanet_transactions.csv"
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Centanet Hong Kong transaction data to CSV."
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help=f"Records per API request. The site currently accepts up to {MAX_PAGE_SIZE}.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.1,
        help="Base delay in seconds between API requests. Default: 1.1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for compatibility. Date-partition mode runs one request at a time.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=600.0,
        help="Seconds to pause and reduce concurrency after a batch still fails. Default: 600",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional debug limit for API pages. Omit for all records.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="First transaction date, YYYY-MM-DD. Default: end-date minus 1095 days.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last transaction date, YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--rebuild-csv-only",
        action="store_true",
        help="Skip downloading and rebuild the CSV from the existing JSONL cache.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing checkpoint/cache files before downloading.",
    )
    return parser.parse_args()


class TqdmProgress:
    def __init__(self, initial: int, total: int | None, workers: int) -> None:
        if tqdm is None:
            raise RuntimeError(
                "tqdm is not installed. Install it with: "
                ".venv/bin/python -m pip install tqdm"
            )
        self.workers = workers
        self.bar = tqdm(
            total=total,
            initial=initial,
            unit="row",
            unit_scale=True,
            dynamic_ncols=True,
            smoothing=0.1,
            desc="Centanet transactions",
        )
        self.done = initial

    def set_total(self, total: int | None) -> None:
        if total is not None and self.bar.total != total:
            self.bar.total = total
            self.bar.refresh()

    def update(self, done: int, total: int | None, next_offset: int) -> None:
        self.set_total(total)
        delta = max(0, done - self.done)
        if delta:
            self.bar.update(delta)
            self.done = done
        self.bar.set_postfix(
            {
                "offset": f"{next_offset:,}",
                "workers": self.workers,
            },
            refresh=True,
        )

    def set_workers(self, workers: int) -> None:
        self.workers = workers

    def finish(self) -> None:
        self.bar.close()


def sidecar_paths(output: Path) -> tuple[Path, Path, Path]:
    cache_dir = Path("cache")
    base_name = output.name
    return (
        cache_dir / f"{base_name}.jsonl",
        cache_dir / f"{base_name}.state.json",
        cache_dir / f"{base_name}.fields.json",
    )


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten dicts; keep lists and complex leaf values as JSON strings."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, name))
        return out

    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False, separators=(",", ":"))}

    return {prefix: value}


def clean_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def parse_api_body(body: str) -> dict[str, Any]:
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected API response type: {type(parsed)!r}")
    return parsed


def post_json_with_urllib(payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(API_URL, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=45) as resp:
        body = resp.read().decode("utf-8")
    return parse_api_body(body)


def post_json_with_curl(payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    if not shutil.which("curl"):
        raise RuntimeError("curl is not available for fallback requests")

    data = json.dumps(payload, ensure_ascii=False)
    cmd = ["curl", "-sS", "-L", "--fail-with-body", API_URL]
    for key, value in headers.items():
        cmd.extend(["-H", f"{key}: {value}"])
    cmd.extend(["--data-raw", data])

    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"curl failed with exit code {completed.returncode}: {detail}")
    return parse_api_body(completed.stdout)


def post_json(payload: dict[str, Any], retries: int = 6) -> dict[str, Any]:
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://hk.centanet.com",
        "Referer": "https://hk.centanet.com/findproperty/en/list/transaction",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Platform": "Web",
    }
    last_error: BaseException | None = None

    for attempt in range(1, retries + 1):
        try:
            try:
                return post_json_with_urllib(payload, headers)
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                return post_json_with_curl(payload, headers)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            last_error = exc
            if attempt == retries:
                raise
            error_text = str(exc)
            if "429" in error_text:
                wait = min(900, 90 * attempt) + random.uniform(15, 45)
            elif "500" in error_text:
                wait = min(300, 30 * attempt) + random.uniform(5, 20)
            else:
                wait = min(60, 2**attempt) + random.uniform(0, 1.5)
            message = (
                f"Request failed at offset={payload.get('offset')} "
                f"(attempt {attempt}/{retries}): {exc}. Sleeping {wait:.1f}s."
            )
            if tqdm is not None:
                tqdm.write(message, file=sys.stderr)
            else:
                print(message, file=sys.stderr, flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Unreachable retry state: {last_error}")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_text(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def request_page(
    offset: int,
    page_size: int,
    post_type: str,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    payload = {
        "postType": post_type,
        "insOrRegDateTimeRange": {
            "start": start_date,
            "end": end_date,
        },
        "sort": "InsOrRegDate",
        "order": "Descending",
        "size": page_size,
        "offset": offset,
    }
    return post_json(payload)


def write_records(
    jsonl: Any,
    records: list[dict[str, Any]],
    partition_id: str,
    offset: int,
    fields: set[str],
) -> int:
    for index, record in enumerate(records):
        row = flatten(record)
        row["_source_partition"] = partition_id
        row["_source_offset"] = offset
        row["_source_index"] = index
        row["_source_api"] = API_URL
        fields.update(row.keys())
        jsonl.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(records)


def count_range(post_type: str, start_date: date, end_date: date) -> int:
    result = request_page(
        offset=0,
        page_size=1,
        post_type=post_type,
        start_date=date_text(start_date),
        end_date=date_text(end_date),
    )
    return int(result.get("count") or 0)


def make_partition(post_type: str, start_date: date, end_date: date, count: int) -> dict[str, Any]:
    start = date_text(start_date)
    end = date_text(end_date)
    return {
        "id": f"{post_type}_{start}_{end}",
        "post_type": post_type,
        "start_date": start,
        "end_date": end,
        "count": count,
        "next_offset": 0,
        "done": count == 0,
        "rows": 0,
    }


def build_partitions(
    start_date: date,
    end_date: date,
    delay: float,
    max_count: int = 10_000,
) -> list[dict[str, Any]]:
    partitions: list[dict[str, Any]] = []

    def split(post_type: str, start: date, end: date) -> None:
        count = count_range(post_type, start, end)
        if delay > 0:
            time.sleep(delay)
        if count <= max_count or start >= end:
            partitions.append(make_partition(post_type, start, end, count))
            return

        mid = start + (end - start) // 2
        split(post_type, start, mid)
        split(post_type, mid + timedelta(days=1), end)

    for post_type in ("Sale", "Rent"):
        split(post_type, start_date, end_date)

    partitions.sort(key=lambda p: (p["start_date"], p["end_date"], p["post_type"]))
    return partitions


def scrape(
    output: Path,
    page_size: int,
    delay: float,
    max_pages: int | None,
    workers: int,
    cooldown: float,
    start_date_arg: str | None,
    end_date_arg: str | None,
) -> None:
    jsonl_path, state_path, fields_path = sidecar_paths(output)
    default_end = date.today()
    default_start = default_end - timedelta(days=1095)
    start_date = parse_date(start_date_arg) if start_date_arg else default_start
    end_date = parse_date(end_date_arg) if end_date_arg else default_end
    state = load_json(state_path, {})
    fields = set(load_json(fields_path, []))

    if state and state.get("mode") != "date_partitions_v1":
        raise RuntimeError(
            "Existing checkpoint was created by the old offset-only scraper. "
            "Run with --fresh once, or use a different output filename."
        )

    if not state:
        print(
            f"Building date partitions for {date_text(start_date)} to {date_text(end_date)}...",
            flush=True,
        )
        partitions = build_partitions(start_date, end_date, delay=delay)
        state = {
            "mode": "date_partitions_v1",
            "start_date": date_text(start_date),
            "end_date": date_text(end_date),
            "count": sum(int(p["count"]) for p in partitions),
            "rows": 0,
            "partitions": partitions,
        }
        save_json(state_path, state)
    else:
        partitions = state["partitions"]

    rows_written = int(state.get("rows") or 0)
    pages_done = 0
    active_workers = 1
    progress = TqdmProgress(
        initial=rows_written,
        total=state.get("count"),
        workers=active_workers,
    )
    progress.update(rows_written, state.get("count"), 0)

    try:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as jsonl:
            for partition in partitions:
                if partition.get("done"):
                    continue

                partition_id = str(partition["id"])
                post_type = str(partition["post_type"])
                part_count = int(partition.get("count") or 0)
                start_text = str(partition["start_date"])
                end_text = str(partition["end_date"])

                while int(partition.get("next_offset") or 0) < part_count:
                    if max_pages is not None and pages_done >= max_pages:
                        save_json(state_path, state)
                        return

                    offset = int(partition.get("next_offset") or 0)
                    try:
                        result = request_page(
                            offset=offset,
                            page_size=page_size,
                            post_type=post_type,
                            start_date=start_text,
                            end_date=end_text,
                        )
                    except Exception as exc:
                        active_workers = 1
                        progress.set_workers(active_workers)
                        if tqdm is not None:
                            tqdm.write(
                                f"Request failed for {partition_id} offset {offset}: {exc}. "
                                f"Cooling down {cooldown:.0f}s."
                            )
                        else:
                            print(
                                f"Request failed for {partition_id} offset {offset}: {exc}. "
                                f"Cooling down {cooldown:.0f}s.",
                                flush=True,
                            )
                        time.sleep(cooldown)
                        continue

                    total = int(result.get("count") or part_count)
                    if total != part_count:
                        partition["count"] = total
                        state["count"] = sum(int(p["count"]) for p in partitions)
                        progress.set_total(state["count"])
                        part_count = total

                    records = result.get("data") or []
                    if not isinstance(records, list):
                        raise RuntimeError(f"Expected list in API data, got {type(records)!r}")

                    if not records:
                        partition["done"] = True
                        save_json(state_path, state)
                        break

                    rows_written += write_records(jsonl, records, partition_id, offset, fields)
                    partition["next_offset"] = offset + len(records)
                    partition["rows"] = int(partition.get("rows") or 0) + len(records)
                    pages_done += 1

                    if int(partition["next_offset"]) >= part_count:
                        partition["done"] = True

                    state["rows"] = rows_written
                    save_json(state_path, state)
                    save_json(fields_path, sorted(fields))

                    jsonl.flush()
                    progress.update(rows_written, state.get("count"), partition["next_offset"])

                    if delay > 0:
                        time.sleep(delay + random.uniform(0, delay / 3))
    finally:
        progress.finish()


def build_csv(output: Path) -> None:
    jsonl_path, _state_path, fields_path = sidecar_paths(output)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing JSONL cache: {jsonl_path}")

    fields = load_json(fields_path, [])
    if not fields:
        raise RuntimeError(f"Missing field list: {fields_path}")

    leading = ["_source_offset", "_source_index", "id", "oldTransactionID", "detailUrl"]
    fieldnames = [name for name in leading if name in fields]
    fieldnames.extend(name for name in fields if name not in fieldnames)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with jsonl_path.open("r", encoding="utf-8") as src, tmp.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        seen_keys: set[str] = set()
        written = 0
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("id") or row.get("oldTransactionID") or "")
            if key:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            writer.writerow({key: clean_for_csv(row.get(key)) for key in fieldnames})
            written += 1

            if line_no % 25000 == 0:
                print(f"Scanned {line_no} JSONL rows, wrote {written} unique CSV rows.", flush=True)

    tmp.replace(output)
    print(f"CSV ready: {output.resolve()}", flush=True)


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser()
    page_size = min(max(1, args.page_size), MAX_PAGE_SIZE)
    jsonl_path, state_path, fields_path = sidecar_paths(output)

    if args.fresh:
        for path in (output, jsonl_path, state_path, fields_path):
            if path.exists():
                path.unlink()

    if not args.rebuild_csv_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        scrape(
            output,
            page_size=page_size,
            delay=args.delay,
            max_pages=args.max_pages,
            workers=args.workers,
            cooldown=args.cooldown,
            start_date_arg=args.start_date,
            end_date_arg=args.end_date,
        )

    build_csv(output)


if __name__ == "__main__":
    main()
