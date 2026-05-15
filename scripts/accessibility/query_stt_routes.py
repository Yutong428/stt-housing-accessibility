#!/usr/bin/env python3
"""
Query Google Routes API driving time from each estate coordinate to each STT point.

Inputs:
  - estate_coordinates.csv
  - stt_points.csv

Output:
  - stt_driving_routes.csv, one row per estate/STT pair

Safety:
  - Dry-run by default.
  - Existing successful or failed rows in the output are reused.
  - Use --run-all to query all missing pairs, or --run --max-requests N for a small test.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_ESTATE_COORDS = "data/processed/estate_coordinates.csv"
DEFAULT_STT_POINTS = "data/processed/stt_points.csv"
DEFAULT_ROUTES_OUTPUT = "cache/stt_driving_routes.csv"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_PRICE_PER_1000_USD = 5.00

OUTPUT_FIELDS = [
    "geocode_query",
    "estate_lat",
    "estate_lng",
    "stt_name",
    "stt_lat",
    "stt_lng",
    "status",
    "duration_seconds",
    "duration_text",
    "distance_meters",
    "route_api",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Google Routes API for STT driving access.")
    parser.add_argument("--estate-coords", default=DEFAULT_ESTATE_COORDS)
    parser.add_argument("--stt-points", default=DEFAULT_STT_POINTS)
    parser.add_argument("--output", default=DEFAULT_ROUTES_OUTPUT)
    parser.add_argument("--api-key", default=None, help="Google Maps API key. Prefer GOOGLE_MAPS_API_KEY.")
    parser.add_argument("--run", action="store_true", help="Actually call Routes API.")
    parser.add_argument("--run-all", action="store_true", help="Call Routes API for all missing pairs.")
    parser.add_argument("--max-requests", type=int, default=0, help="Maximum new API requests with --run.")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls. Default: 0.1")
    parser.add_argument("--price-per-1000", type=float, default=ROUTES_PRICE_PER_1000_USD)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["geocode_query"], row["stt_name"])


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("geocode_query", "")
            stt = row.get("stt_name", "")
            if query and stt:
                existing[(query, stt)] = row
    return existing


def write_routes(path: Path, rows: dict[tuple[str, str], dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({field: rows[key].get(field, "") for field in OUTPUT_FIELDS})
    tmp.replace(path)


def build_pairs(estates: list[dict[str, str]], stt_points: list[dict[str, str]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for estate in estates:
        for stt in stt_points:
            pairs.append(
                {
                    "geocode_query": estate["geocode_query"],
                    "estate_lat": estate["estate_lat"],
                    "estate_lng": estate["estate_lng"],
                    "stt_name": stt["stt_name"],
                    "stt_lat": stt["stt_lat"],
                    "stt_lng": stt["stt_lng"],
                }
            )
    return pairs


def parse_duration_seconds(duration: str) -> str:
    if not duration:
        return ""
    return duration.rstrip("s")


def duration_text(seconds_text: str) -> str:
    if not seconds_text:
        return ""
    try:
        seconds = int(float(seconds_text))
    except ValueError:
        return ""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def compute_route(pair: dict[str, str], api_key: str) -> dict[str, str]:
    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(pair["estate_lat"]),
                    "longitude": float(pair["estate_lng"]),
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(pair["stt_lat"]),
                    "longitude": float(pair["stt_lng"]),
                }
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_UNAWARE",
        "computeAlternativeRoutes": False,
        "languageCode": "zh-HK",
        "regionCode": "HK",
    }
    request = Request(
        ROUTES_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {**pair, "status": "ERROR", "route_api": "Routes API ComputeRoutes", "error_message": str(exc)}

    routes = payload.get("routes") or []
    if not routes:
        return {
            **pair,
            "status": "ZERO_RESULTS",
            "route_api": "Routes API ComputeRoutes",
            "error_message": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        }

    route = routes[0]
    seconds_text = parse_duration_seconds(str(route.get("duration", "")))
    return {
        **pair,
        "status": "OK",
        "duration_seconds": seconds_text,
        "duration_text": duration_text(seconds_text),
        "distance_meters": str(route.get("distanceMeters", "")),
        "route_api": "Routes API ComputeRoutes",
        "error_message": "",
    }


def main() -> None:
    args = parse_args()
    estates = read_csv(Path(args.estate_coords).expanduser())
    stt_points = read_csv(Path(args.stt_points).expanduser())
    output_path = Path(args.output).expanduser()
    existing = load_existing(output_path)
    pairs = build_pairs(estates, stt_points)
    missing = [pair for pair in pairs if pair_key(pair) not in existing]

    should_run = args.run or args.run_all
    planned = len(missing) if args.run_all else min(args.max_requests, len(missing))
    estimated_cost = planned / 1000 * args.price_per_1000
    print(f"Estate coordinates: {len(estates):,}")
    print(f"STT points: {len(stt_points):,}")
    print(f"Total route pairs: {len(pairs):,}")
    print(f"Already cached route pairs: {len(existing):,}")
    print(f"Missing route pairs: {len(missing):,}")
    print(f"Planned API requests this run: {planned:,}")
    print(f"Local cost estimate before free credits/quota: ${estimated_cost:.4f} USD")

    if not should_run:
        print("Dry run only. No Routes API calls made. Pass --run-all to query all missing pairs.")
        return
    if not args.run_all and args.max_requests <= 0:
        raise ValueError("Use --max-requests N with --run, or pass --run-all.")

    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set GOOGLE_MAPS_API_KEY or pass --api-key.")

    rows = dict(existing)
    for index, pair in enumerate(missing[:planned], start=1):
        print(f"[{index}/{planned}] {pair['geocode_query']} -> {pair['stt_name']}", flush=True)
        result = compute_route(pair, api_key)
        rows[pair_key(pair)] = result
        write_routes(output_path, rows)
        if args.delay > 0 and index < planned:
            time.sleep(args.delay)

    print(f"Wrote routes table: {output_path.resolve()}")


if __name__ == "__main__":
    main()
