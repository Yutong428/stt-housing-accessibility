#!/usr/bin/env python3
"""
Compute each estate's nearest STT point by haversine/euclidean surface distance,
then broadcast that link back to the transaction-level main table.

This does not call any external API.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_ESTATE_COORDS = "data/processed/estate_coordinates.csv"
DEFAULT_STT_POINTS = "data/processed/stt_points.csv"
DEFAULT_MAIN = "data/processed/centanet_transactions_stt_accessibility.csv"
DEFAULT_LINK_OUTPUT = "data/processed/estate_nearest_stt_by_euclidean.csv"
DEFAULT_MAIN_OUTPUT = "data/processed/centanet_transactions_stt_accessibility_final.csv"
EARTH_RADIUS_METERS = 6_371_000

LINK_FIELDS = [
    "geocode_query",
    "estate_lat",
    "estate_lng",
    "nearest_euclidean_stt_name",
    "nearest_euclidean_stt_lat",
    "nearest_euclidean_stt_lng",
    "nearest_euclidean_distance_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link estates to nearest STT by euclidean distance.")
    parser.add_argument("--estate-coords", default=DEFAULT_ESTATE_COORDS)
    parser.add_argument("--stt-points", default=DEFAULT_STT_POINTS)
    parser.add_argument("--main", default=DEFAULT_MAIN)
    parser.add_argument("--link-output", default=DEFAULT_LINK_OUTPUT)
    parser.add_argument("--main-output", default=DEFAULT_MAIN_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return EARTH_RADIUS_METERS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_links(estates: list[dict[str, str]], stt_points: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    links: dict[str, dict[str, str]] = {}
    for estate in estates:
        estate_lat = float(estate["estate_lat"])
        estate_lng = float(estate["estate_lng"])
        best_stt: dict[str, str] | None = None
        best_distance: float | None = None

        for stt in stt_points:
            distance = haversine_meters(
                estate_lat,
                estate_lng,
                float(stt["stt_lat"]),
                float(stt["stt_lng"]),
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_stt = stt

        if best_stt is None or best_distance is None:
            continue

        links[estate["geocode_query"]] = {
            "geocode_query": estate["geocode_query"],
            "estate_lat": estate["estate_lat"],
            "estate_lng": estate["estate_lng"],
            "nearest_euclidean_stt_name": best_stt["stt_name"],
            "nearest_euclidean_stt_lat": best_stt["stt_lat"],
            "nearest_euclidean_stt_lng": best_stt["stt_lng"],
            "nearest_euclidean_distance_km": str(best_distance / 1000),
        }
    return links


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
    estates = read_csv(Path(args.estate_coords).expanduser())
    stt_points = read_csv(Path(args.stt_points).expanduser())
    links = compute_links(estates, stt_points)
    write_links(Path(args.link_output).expanduser(), links)
    rows = broadcast_to_main(Path(args.main).expanduser(), Path(args.main_output).expanduser(), links)
    print(f"Wrote {len(links):,} euclidean links to {Path(args.link_output).resolve()}")
    print(f"Wrote {rows:,} transaction rows to {Path(args.main_output).resolve()}")


if __name__ == "__main__":
    main()
