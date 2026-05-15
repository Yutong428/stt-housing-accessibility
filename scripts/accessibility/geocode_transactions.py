#!/usr/bin/env python3
"""
Geocode unique estate-level queries from the prepared Centanet target-area CSV.

Safety defaults:
  - Dry-run by default: no API calls unless --run is passed.
  - Unique geocode_query values are queried once and cached in a separate CSV.
  - Existing cache rows are reused, so reruns do not spend quota again.
  - --max-requests limits paid API calls per run.

API key:
  export GOOGLE_MAPS_API_KEY="your_key"
  .venv/bin/python scripts/accessibility/geocode_transactions.py --run-all

The key can also be passed with --api-key, but environment variables are safer
because they avoid writing secrets into shell history and files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


DEFAULT_INPUT = "data/processed/centanet_transactions_sale_target_areas.csv"
DEFAULT_CACHE = "cache/geocode_cache.csv"
DEFAULT_OUTPUT = "data/processed/centanet_transactions_sale_target_areas_geocoded.csv"

GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GEOCODING_PRICE_PER_1000_USD = 5.00
FREE_MONTHLY_REQUESTS = 10_000

CACHE_FIELDS = [
    "geocode_query",
    "status",
    "formatted_address",
    "place_id",
    "location_type",
    "lat",
    "lng",
    "viewport_northeast_lat",
    "viewport_northeast_lng",
    "viewport_southwest_lat",
    "viewport_southwest_lng",
    "partial_match",
    "types",
    "raw_result_json",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geocode unique geocode_query values.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV. Default: {DEFAULT_INPUT}")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help=f"Geocode cache CSV. Default: {DEFAULT_CACHE}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Broadcast output CSV. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--api-key", default=None, help="Google Maps API key. Prefer GOOGLE_MAPS_API_KEY.")
    parser.add_argument("--run", action="store_true", help="Actually call the Google Geocoding API.")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum new API requests in this run. Required with --run unless --run-all is used.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Call the API for all missing unique queries.",
    )
    parser.add_argument("--delay", type=float, default=0.15, help="Delay between API calls. Default: 0.15")
    parser.add_argument(
        "--free-monthly-requests",
        type=int,
        default=FREE_MONTHLY_REQUESTS,
        help="Free monthly Geocoding request estimate for local cost reporting. Default: 10000",
    )
    parser.add_argument(
        "--price-per-1000",
        type=float,
        default=GEOCODING_PRICE_PER_1000_USD,
        help="Geocoding price estimate in USD per 1000 requests. Default: 5.00",
    )
    return parser.parse_args()


def read_unique_queries(input_path: Path) -> list[str]:
    queries: set[str] = set()
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        if "geocode_query" not in reader.fieldnames:
            raise ValueError("Input CSV must contain geocode_query")
        for row in reader:
            query = row.get("geocode_query", "").strip()
            if query:
                queries.add(query)
    return sorted(queries)


def read_cache(cache_path: Path) -> dict[str, dict[str, str]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, str]] = {}
    with cache_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("geocode_query", "").strip()
            if query:
                cache[query] = row
    return cache


def write_cache(cache_path: Path, cache: dict[str, dict[str, str]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for query in sorted(cache):
            row = {field: cache[query].get(field, "") for field in CACHE_FIELDS}
            row["geocode_query"] = query
            writer.writerow(row)
    tmp.replace(cache_path)


def estimate_cost(requests: int, free_monthly_requests: int, price_per_1000: float) -> float:
    billable = max(0, requests - free_monthly_requests)
    return billable / 1000 * price_per_1000


def geocode_query(query: str, api_key: str) -> dict[str, str]:
    params = {
        "address": query,
        "region": "hk",
        "language": "zh-HK",
        "key": api_key,
    }
    url = f"{GEOCODING_URL}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))

    status = str(payload.get("status", ""))
    error_message = str(payload.get("error_message", ""))
    results = payload.get("results") or []
    if not results:
        return {
            "geocode_query": query,
            "status": status,
            "error_message": error_message,
            "raw_result_json": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }

    result = results[0]
    geometry = result.get("geometry") or {}
    location = geometry.get("location") or {}
    viewport = geometry.get("viewport") or {}
    northeast = viewport.get("northeast") or {}
    southwest = viewport.get("southwest") or {}
    return {
        "geocode_query": query,
        "status": status,
        "formatted_address": str(result.get("formatted_address", "")),
        "place_id": str(result.get("place_id", "")),
        "location_type": str(geometry.get("location_type", "")),
        "lat": str(location.get("lat", "")),
        "lng": str(location.get("lng", "")),
        "viewport_northeast_lat": str(northeast.get("lat", "")),
        "viewport_northeast_lng": str(northeast.get("lng", "")),
        "viewport_southwest_lat": str(southwest.get("lat", "")),
        "viewport_southwest_lng": str(southwest.get("lng", "")),
        "partial_match": str(result.get("partial_match", "")),
        "types": json.dumps(result.get("types", []), ensure_ascii=False, separators=(",", ":")),
        "raw_result_json": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        "error_message": error_message,
    }


def update_cache(
    queries: list[str],
    cache: dict[str, dict[str, str]],
    cache_path: Path,
    api_key: str,
    max_requests: int,
    delay: float,
) -> int:
    missing = [query for query in queries if query not in cache]
    to_query = missing[:max_requests]
    for index, query in enumerate(to_query, start=1):
        print(f"[{index}/{len(to_query)}] Geocoding: {query}", flush=True)
        cache[query] = geocode_query(query, api_key)
        write_cache(cache_path, cache)
        if delay > 0 and index < len(to_query):
            time.sleep(delay)
    return len(to_query)


def broadcast_cache(input_path: Path, output_path: Path, cache: dict[str, dict[str, str]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    geocode_fields = [field for field in CACHE_FIELDS if field != "geocode_query" and field != "raw_result_json"]
    output_geocode_fields = [f"geocode_{field}" for field in geocode_fields]

    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        fieldnames = list(reader.fieldnames) + output_geocode_fields

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            row_count = 0
            for row in reader:
                cached = cache.get(row.get("geocode_query", "").strip(), {})
                for source_field, output_field in zip(geocode_fields, output_geocode_fields):
                    row[output_field] = cached.get(source_field, "")
                writer.writerow(row)
                row_count += 1
    return row_count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    cache_path = Path(args.cache).expanduser()
    output_path = Path(args.output).expanduser()

    queries = read_unique_queries(input_path)
    cache = read_cache(cache_path)
    missing = [query for query in queries if query not in cache]

    should_run = args.run or args.run_all
    planned_requests = len(missing) if args.run_all else args.max_requests
    planned_requests = min(planned_requests, len(missing))
    estimated_cost = estimate_cost(
        planned_requests,
        args.free_monthly_requests,
        args.price_per_1000,
    )
    free_remaining_after_plan = max(0, args.free_monthly_requests - planned_requests)

    print(f"Rows source: {input_path}")
    print(f"Unique geocode_query values: {len(queries):,}")
    print(f"Already cached: {len(cache):,}")
    print(f"Missing unique queries: {len(missing):,}")
    print(f"Planned API requests this run: {planned_requests:,}")
    print(
        "Local cost estimate for this run: "
        f"${estimated_cost:.4f} USD "
        f"(assuming {args.free_monthly_requests:,} free monthly requests and "
        f"${args.price_per_1000:.2f}/1000 after that)"
    )
    print(f"Estimated free monthly requests remaining after this run: {free_remaining_after_plan:,}")

    if not should_run:
        print("Dry run only. No API calls made. Pass --run-all to call the API for all missing queries.")
        return

    if not args.run_all and args.max_requests <= 0:
        raise ValueError("Use --max-requests N with --run, or pass --run-all.")

    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key.")

    made_requests = update_cache(
        queries=queries,
        cache=cache,
        cache_path=cache_path,
        api_key=api_key,
        max_requests=planned_requests,
        delay=args.delay,
    )
    print(f"New API requests made: {made_requests:,}")

    refreshed_cache = read_cache(cache_path)
    row_count = broadcast_cache(input_path, output_path, refreshed_cache)
    print(f"Wrote geocode cache: {cache_path.resolve()}")
    print(f"Wrote broadcast output with {row_count:,} rows: {output_path.resolve()}")


if __name__ == "__main__":
    main()
