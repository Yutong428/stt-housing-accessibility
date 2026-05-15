#!/usr/bin/env python3
"""
Compute each estate's nearest STT point by Google Routes driving time, then
broadcast that link back to the transaction-level main table.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ROUTES = "cache/stt_driving_routes.csv"
DEFAULT_MAIN = "data/processed/centanet_transactions_sale_target_areas_geocoded.csv"
DEFAULT_LINK_OUTPUT = "data/processed/estate_nearest_stt_by_driving.csv"
DEFAULT_MAIN_OUTPUT = "data/processed/centanet_transactions_stt_accessibility.csv"

LINK_FIELDS = [
    "geocode_query",
    "estate_lat",
    "estate_lng",
    "nearest_driving_stt_name",
    "nearest_driving_stt_lat",
    "nearest_driving_stt_lng",
    "nearest_driving_time_min",
    "nearest_driving_route_distance_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link estates to nearest STT by driving time.")
    parser.add_argument("--routes", default=DEFAULT_ROUTES)
    parser.add_argument("--main", default=DEFAULT_MAIN)
    parser.add_argument("--link-output", default=DEFAULT_LINK_OUTPUT)
    parser.add_argument("--main-output", default=DEFAULT_MAIN_OUTPUT)
    return parser.parse_args()


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_links(routes_path: Path) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    with routes_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            query = row.get("geocode_query", "")
            duration = to_float(row.get("duration_seconds", ""))
            if not query or duration is None:
                continue
            current = best.get(query)
            current_duration = to_float(current.get("_duration_seconds", "")) if current else None
            if current is not None and current_duration is not None and duration >= current_duration:
                continue
            best[query] = {
                "geocode_query": query,
                "estate_lat": row.get("estate_lat", ""),
                "estate_lng": row.get("estate_lng", ""),
                "nearest_driving_stt_name": row.get("stt_name", ""),
                "nearest_driving_stt_lat": row.get("stt_lat", ""),
                "nearest_driving_stt_lng": row.get("stt_lng", ""),
                "nearest_driving_time_min": str(duration / 60),
                "nearest_driving_route_distance_km": str((to_float(row.get("distance_meters", "")) or 0) / 1000),
                "_duration_seconds": row.get("duration_seconds", ""),
            }
    return best


def write_links(path: Path, links: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LINK_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for query in sorted(links):
            writer.writerow(links[query])


def broadcast_to_main(main_path: Path, output_path: Path, links: dict[str, dict[str, str]]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    broadcast_fields = [field for field in LINK_FIELDS if field != "geocode_query"]
    with main_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Main CSV has no header: {main_path}")
        fieldnames = list(reader.fieldnames)
        for field in broadcast_fields:
            if field not in fieldnames:
                fieldnames.append(field)

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            rows = 0
            for row in reader:
                link = links.get(row.get("geocode_query", ""))
                if link:
                    for field in broadcast_fields:
                        row[field] = link.get(field, "")
                writer.writerow(row)
                rows += 1
    return rows


def main() -> None:
    args = parse_args()
    links = compute_links(Path(args.routes).expanduser())
    write_links(Path(args.link_output).expanduser(), links)
    rows = broadcast_to_main(Path(args.main).expanduser(), Path(args.main_output).expanduser(), links)
    print(f"Wrote {len(links):,} driving links to {Path(args.link_output).resolve()}")
    print(f"Wrote {rows:,} transaction rows to {Path(args.main_output).resolve()}")


if __name__ == "__main__":
    main()
