#!/usr/bin/env python3
"""
Create estate coordinate and San Tin Technopole/STT destination coordinate CSVs.

Inputs:
  - geocode_cache.csv from geocode_transactions.py

Outputs:
  - estate_coordinates.csv: unique geocode_query with lat/lng
  - stt_points.csv: fixed STT internal points with name/lat/lng
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_GEOCODE_CACHE = "cache/geocode_cache.csv"
DEFAULT_ESTATE_COORDS = "data/processed/estate_coordinates.csv"
DEFAULT_STT_POINTS = "data/processed/stt_points.csv"

STT_POINTS = [
    {
        "stt_name": "Proposed HSITP Station",
        "stt_lat": "22.522195",
        "stt_lng": "114.080236",
    },
    {
        "stt_name": "Proposed San Tin Station",
        "stt_lat": "22.486925",
        "stt_lng": "114.072244",
    },
    {
        "stt_name": "Proposed Railway Station and Transport Interchange near San Tin Interchange",
        "stt_lat": "22.502880",
        "stt_lng": "114.078952",
    },
    {
        "stt_name": "Proposed Southeastern Area Transport Interchange",
        "stt_lat": "22.490287",
        "stt_lng": "114.087515",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare estate and STT coordinate CSVs.")
    parser.add_argument("--geocode-cache", default=DEFAULT_GEOCODE_CACHE)
    parser.add_argument("--estate-output", default=DEFAULT_ESTATE_COORDS)
    parser.add_argument("--stt-output", default=DEFAULT_STT_POINTS)
    return parser.parse_args()


def is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def write_estate_coordinates(geocode_cache_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    row_count = 0

    with geocode_cache_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        required = ["geocode_query", "status", "lat", "lng"]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Missing required geocode cache columns: " + ", ".join(missing))

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=["geocode_query", "estate_lat", "estate_lng"])
            writer.writeheader()
            for row in reader:
                query = row.get("geocode_query", "").strip()
                lat = row.get("lat", "").strip()
                lng = row.get("lng", "").strip()
                if not query or query in seen:
                    continue
                if row.get("status") != "OK" or not is_number(lat) or not is_number(lng):
                    continue
                seen.add(query)
                writer.writerow({"geocode_query": query, "estate_lat": lat, "estate_lng": lng})
                row_count += 1

    return row_count


def write_stt_points(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["stt_name", "stt_lat", "stt_lng"])
        writer.writeheader()
        writer.writerows(STT_POINTS)
    return len(STT_POINTS)


def main() -> None:
    args = parse_args()
    estate_rows = write_estate_coordinates(
        Path(args.geocode_cache).expanduser(),
        Path(args.estate_output).expanduser(),
    )
    stt_rows = write_stt_points(Path(args.stt_output).expanduser())
    print(f"Wrote {estate_rows:,} estate coordinates to {Path(args.estate_output).resolve()}")
    print(f"Wrote {stt_rows:,} STT points to {Path(args.stt_output).resolve()}")


if __name__ == "__main__":
    main()
