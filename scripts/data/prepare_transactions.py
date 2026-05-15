#!/usr/bin/env python3
"""
Create the working transaction dataset for the selected New Territories areas.

The crawler and raw crawler output are intentionally left untouched. This script
only reads the raw CSV and writes a separate filtered CSV with all original
columns transformed into the analysis schema.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_INPUT = "data/raw/centanet_transactions.csv"
DEFAULT_OUTPUT = "data/processed/centanet_transactions_sale_target_areas.csv"

TARGET_WEBSCOPE_VALUES = {
    "元朗",
    "上水 | 粉嶺 | 古洞",
    "天水圍",
    "錦繡 | 加州 | 葡萄園",
    "屯門",
    "大埔",
}

OUTPUT_FIELDS = [
    "transaction_id",
    "old_transaction_id",
    "source_url",
    "trans_theme",
    "post_type",
    "first_or_second_hand",
    "transaction_date",
    "registration_date",
    "month",
    "post_ozp",
    "price",
    "log_price",
    "saleable_area",
    "log_area",
    "price_psf",
    "log_price_psf",
    "gross_area",
    "gross_price_psf",
    "district_name",
    "admin_district",
    "territory",
    "market_area",
    "hma",
    "web_scope",
    "estate_name",
    "big_estate_name",
    "building_name",
    "building_group_id",
    "building_group_name",
    "building_group_url",
    "full_address",
    "area_address",
    "street_address",
    "floor_raw",
    "floor_group",
    "unit",
    "bedroom_count",
    "direction",
    "op_year",
    "building_age",
    "special_case_value",
    "special_case_remark",
    "data_source",
    "old_data_source",
    "show_map",
    "geocode_query",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter Centanet transactions to sale records in selected "
            "scope.webScope areas and output the analysis schema."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Raw crawler CSV path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Filtered output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def log_value(value: str | None) -> str:
    number = parse_number(value)
    if number is None or number <= 0:
        return ""
    return str(math.log(number))


def date_part(value: str | None) -> str:
    if not value:
        return ""
    return value[:10]


def month_part(value: str | None) -> str:
    clean_date = date_part(value)
    if len(clean_date) < 7:
        return ""
    return clean_date[:7]


def parse_year(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\d{4}", value)
    if not match:
        return None
    return int(match.group(0))


def transaction_year(value: str | None) -> int | None:
    clean_date = date_part(value)
    if len(clean_date) < 4:
        return None
    try:
        return int(clean_date[:4])
    except ValueError:
        return None


def building_age(op_year: str | None, transaction_date: str | None) -> str:
    op = parse_year(op_year)
    trans = transaction_year(transaction_date)
    if op is None or trans is None:
        return ""
    return str(trans - op)


def floor_group(value: str | None) -> str:
    if not value:
        return ""
    if "低" in value or "Lower" in value:
        return "low"
    if "中" in value or "Middle" in value:
        return "middle"
    if "高" in value or "Upper" in value:
        return "high"
    return ""


def transform_row(row: dict[str, str]) -> dict[str, str]:
    transaction_date = date_part(row.get("insDate"))
    estate_name = row.get("estateName", "")
    big_estate_name = row.get("bigEstateName", "")
    admin_district = row.get("scope.db", "")
    geocode_parts = [estate_name]
    if big_estate_name:
        geocode_parts.append(big_estate_name)
    geocode_parts.extend([admin_district, "香港"])

    output = {
        "transaction_id": row.get("id", ""),
        "old_transaction_id": row.get("oldTransactionID", ""),
        "source_url": row.get("detailUrl", ""),
        "trans_theme": row.get("transTheme", ""),
        "post_type": row.get("postType", ""),
        "first_or_second_hand": row.get("firstOrSecondHand", ""),
        "transaction_date": transaction_date,
        "registration_date": date_part(row.get("regDate")),
        "month": month_part(transaction_date),
        "post_ozp": "",
        "price": row.get("transactionPrice", ""),
        "log_price": log_value(row.get("transactionPrice")),
        "saleable_area": row.get("nArea", ""),
        "log_area": log_value(row.get("nArea")),
        "price_psf": row.get("nUnitPrice", ""),
        "log_price_psf": log_value(row.get("nUnitPrice")),
        "gross_area": row.get("gArea", ""),
        "gross_price_psf": row.get("gUnitPrice", ""),
        "district_name": row.get("districtName", ""),
        "admin_district": row.get("scope.db", ""),
        "territory": row.get("scope.terr", ""),
        "market_area": row.get("scope.scp_mkt", ""),
        "hma": row.get("scope.hma", ""),
        "web_scope": row.get("scope.webScope", ""),
        "estate_name": estate_name,
        "big_estate_name": big_estate_name,
        "building_name": row.get("buildingName", ""),
        "building_group_id": row.get("bldgGrp.bldgGrpId", ""),
        "building_group_name": row.get("bldgGrp.bldgGrpName", ""),
        "building_group_url": row.get("bldgGrp.url", ""),
        "full_address": row.get("displayText.addr.line1", ""),
        "area_address": row.get("displayText.addr.line5", ""),
        "street_address": row.get("address", ""),
        "floor_raw": row.get("yAxis", ""),
        "floor_group": floor_group(row.get("yAxis")),
        "unit": row.get("xAxis", ""),
        "bedroom_count": row.get("bedroomCount", ""),
        "direction": row.get("direction", ""),
        "op_year": row.get("opYear", ""),
        "building_age": building_age(row.get("opYear"), transaction_date),
        "special_case_value": row.get("specialCase.value", ""),
        "special_case_remark": row.get("specialCase.remark", ""),
        "data_source": row.get("dataSource", ""),
        "old_data_source": row.get("oldDataSource", ""),
        "show_map": row.get("showMap", ""),
    }
    output["geocode_query"] = ", ".join(part for part in geocode_parts if part)
    return output


def filter_sale_target_areas(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")

        required = [
            "postType",
            "scope.webScope",
            "id",
            "transactionPrice",
            "insDate",
            "nArea",
            "nUnitPrice",
        ]
        missing = [field for field in required if field not in reader.fieldnames]
        if missing:
            raise ValueError("Missing required columns in input CSV: " + ", ".join(missing))

        with output_path.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            writer.writeheader()

            row_count = 0
            for row in reader:
                if row.get("postType") != "S":
                    continue
                if row.get("scope.webScope") not in TARGET_WEBSCOPE_VALUES:
                    continue
                if not row.get("estateName"):
                    continue
                writer.writerow(transform_row(row))
                row_count += 1

    return row_count


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    row_count = filter_sale_target_areas(input_path, output_path)
    print(f"Wrote {row_count:,} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
