#!/usr/bin/env python3
"""
Create STT accessibility heatmap maps using adjusted estate price growth.

Outputs:
  - estate-level hedonic-residual growth data
  - coordinate and extent summaries
  - Google route-based and Euclidean distance PDF maps
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))

import contextily as cx
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from shapely.geometry import LineString, box


matplotlib.use("Agg")


DEFAULT_INPUT = "data/processed/analysis_sample_used.csv"
DEFAULT_STT_POINTS = "data/processed/stt_points.csv"
DEFAULT_OUTPUT_DIR = "results"
SHOCK_DATE = pd.Timestamp("2024-03-08")
PRE_START = pd.Timestamp("2023-03-08")
PRE_END = pd.Timestamp("2024-03-07")
POST_START = pd.Timestamp("2024-03-08")
POST_END = pd.Timestamp("2025-03-07")
HK_LAT_MIN = 22.15
HK_LAT_MAX = 22.60
HK_LNG_MIN = 113.80
HK_LNG_MAX = 114.45
ANCHORS = {
    "north_lat": 22.526714,
    "west_lng": 114.054891,
    "east_lng": 114.099278,
    "south_lat": 22.479418,
}
STT_COLORS = {
    "Proposed HSITP Station": "#39FF14",
    "Proposed San Tin Station": "#00E5FF",
    "Proposed Railway Station and Transport Interchange near San Tin Interchange": "#FFFF00",
    "Proposed Southeastern Area Transport Interchange": "#FF4DFF",
}


REQUIRED_COLUMNS = [
    "estate_id",
    "transaction_date",
    "month",
    "log_price_psf",
    "log_area",
    "floor_group",
    "building_age",
    "estate_lat",
    "estate_lng",
    "nearest_driving_stt_name",
    "nearest_driving_stt_lat",
    "nearest_driving_stt_lng",
    "nearest_driving_time_min",
    "nearest_driving_route_distance_km",
    "nearest_euclidean_stt_name",
    "nearest_euclidean_stt_lat",
    "nearest_euclidean_stt_lng",
    "nearest_euclidean_distance_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create STT adjusted-growth heatmap maps.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Transaction-level analysis CSV. Default: {DEFAULT_INPUT}")
    parser.add_argument("--stt-points", default=DEFAULT_STT_POINTS, help=f"STT node CSV. Default: {DEFAULT_STT_POINTS}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output root directory. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=[6, 10, 20],
        help="Minimum n_pre and n_post thresholds to run. Default: 6 10 20",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def clean_string(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    out = series.astype("string").fillna(missing_label).str.strip()
    return out.mask(out == "", missing_label)


def normalize_node_name(series: pd.Series) -> pd.Series:
    return clean_string(series, "").str.lower().str.replace(r"\s+", " ", regex=True)


def to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    require_columns(df, REQUIRED_COLUMNS)

    df["estate_id"] = clean_string(df["estate_id"])
    if "estate_name" not in df.columns:
        df["estate_name"] = df["estate_id"]
    df["estate_name"] = clean_string(df["estate_name"])
    if "first_or_second_hand" not in df.columns:
        df["first_or_second_hand"] = "Missing"

    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    month_raw = df["month"].astype("string")
    df["month"] = pd.to_datetime(month_raw, errors="coerce")
    month_from_ym = pd.to_datetime(month_raw + "-01", errors="coerce")
    df["month"] = df["month"].fillna(month_from_ym)
    df["month_id"] = df["month"].dt.strftime("%Y-%m")

    numeric_columns = [
        "log_price_psf",
        "log_area",
        "building_age",
        "estate_lat",
        "estate_lng",
        "nearest_driving_stt_lat",
        "nearest_driving_stt_lng",
        "nearest_driving_time_min",
        "nearest_driving_route_distance_km",
        "nearest_euclidean_stt_lat",
        "nearest_euclidean_stt_lng",
        "nearest_euclidean_distance_km",
    ]
    to_numeric(df, numeric_columns)
    df["floor_group"] = clean_string(df["floor_group"], "Missing")
    df["first_or_second_hand"] = clean_string(df["first_or_second_hand"], "Missing")
    df["nearest_driving_stt_name"] = clean_string(df["nearest_driving_stt_name"])
    df["nearest_euclidean_stt_name"] = clean_string(df["nearest_euclidean_stt_name"])
    df["detour_ratio"] = df["nearest_driving_route_distance_km"] / df["nearest_euclidean_distance_km"]

    return df


def load_stt_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    require_columns(df, ["stt_name", "stt_lat", "stt_lng"])
    df["stt_name"] = clean_string(df["stt_name"])
    df["stt_lat"] = pd.to_numeric(df["stt_lat"], errors="coerce")
    df["stt_lng"] = pd.to_numeric(df["stt_lng"], errors="coerce")
    df = df.dropna(subset=["stt_name", "stt_lat", "stt_lng"]).drop_duplicates("stt_name").reset_index(drop=True)
    fallback_colors = ["#39FF14", "#00E5FF", "#FFFF00", "#FF4DFF", "#FF7F00", "#66FF66"]
    df["stt_color"] = [
        STT_COLORS.get(str(name), fallback_colors[i % len(fallback_colors)])
        for i, name in enumerate(df["stt_name"])
    ]
    return df


def hedonic_window_sample(df: pd.DataFrame) -> pd.DataFrame:
    in_window = df["transaction_date"].between(PRE_START, POST_END, inclusive="both")
    sample = df.loc[in_window].copy()
    needed = [
        "log_price_psf",
        "log_area",
        "building_age",
        "month_id",
        "transaction_date",
        "estate_id",
        "estate_lat",
        "estate_lng",
    ]
    sample = sample.dropna(subset=needed).copy()
    sample = sample.loc[sample["estate_id"].astype(str).str.len() > 0].copy()
    sample["period"] = np.where(sample["transaction_date"] < SHOCK_DATE, "pre", "post")
    return sample


def fit_hedonic_residuals(sample: pd.DataFrame) -> pd.DataFrame:
    formula = "log_price_psf ~ log_area + C(floor_group) + building_age + C(month_id)"
    if sample["first_or_second_hand"].nunique(dropna=True) > 1:
        formula += " + C(first_or_second_hand)"
    model = smf.ols(formula=formula, data=sample).fit()
    out = sample.copy()
    out["hedonic_predicted"] = model.predict(out)
    out["hedonic_residual"] = out["log_price_psf"] - out["hedonic_predicted"]
    return out


def first_nonmissing(series: pd.Series) -> Any:
    nonmissing = series.dropna()
    if nonmissing.empty:
        return pd.NA
    return nonmissing.iloc[0]


def build_estate_growth(residuals: pd.DataFrame, min_period_n: int) -> pd.DataFrame:
    static_columns = [
        "estate_name",
        "estate_lat",
        "estate_lng",
        "nearest_driving_stt_name",
        "nearest_driving_stt_lat",
        "nearest_driving_stt_lng",
        "nearest_driving_time_min",
        "nearest_driving_route_distance_km",
        "nearest_euclidean_stt_name",
        "nearest_euclidean_stt_lat",
        "nearest_euclidean_stt_lng",
        "nearest_euclidean_distance_km",
        "detour_ratio",
    ]
    static = residuals.groupby("estate_id", observed=True)[static_columns].agg(first_nonmissing)
    means = residuals.pivot_table(
        index="estate_id",
        columns="period",
        values="hedonic_residual",
        aggfunc=["mean", "count"],
        observed=True,
    )
    means.columns = [f"{period}_{stat}" for stat, period in means.columns]
    estate = static.join(means, how="inner").reset_index()
    estate = estate.rename(
        columns={
            "pre_mean": "pre_resid_mean",
            "post_mean": "post_resid_mean",
            "pre_count": "n_pre",
            "post_count": "n_post",
        }
    )
    for column in ["n_pre", "n_post"]:
        estate[column] = estate[column].fillna(0).astype(int)
    estate["n_total"] = estate["n_pre"] + estate["n_post"]
    estate["adjusted_growth_log"] = estate["post_resid_mean"] - estate["pre_resid_mean"]
    estate["adjusted_growth_pct"] = 100 * (np.exp(estate["adjusted_growth_log"]) - 1)

    driving_norm = normalize_node_name(estate["nearest_driving_stt_name"])
    euclidean_norm = normalize_node_name(estate["nearest_euclidean_stt_name"])
    estate["driving_node_differs_from_euclidean"] = driving_norm != euclidean_norm

    keep = estate.loc[(estate["n_pre"] >= min_period_n) & (estate["n_post"] >= min_period_n)].copy()
    ordered_columns = [
        "estate_id",
        "estate_name",
        "estate_lat",
        "estate_lng",
        "n_pre",
        "n_post",
        "n_total",
        "pre_resid_mean",
        "post_resid_mean",
        "adjusted_growth_log",
        "adjusted_growth_pct",
        "nearest_driving_stt_name",
        "nearest_driving_stt_lat",
        "nearest_driving_stt_lng",
        "nearest_driving_time_min",
        "nearest_driving_route_distance_km",
        "nearest_euclidean_stt_name",
        "nearest_euclidean_stt_lat",
        "nearest_euclidean_stt_lng",
        "nearest_euclidean_distance_km",
        "detour_ratio",
        "driving_node_differs_from_euclidean",
    ]
    return keep[ordered_columns].sort_values("estate_id").reset_index(drop=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or (list(rows[0].keys()) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coordinate_check(estate: pd.DataFrame, stt_points: pd.DataFrame) -> dict[str, Any]:
    missing_estate = estate["estate_lat"].isna() | estate["estate_lng"].isna()
    missing_google = estate["nearest_driving_stt_lat"].isna() | estate["nearest_driving_stt_lng"].isna()
    missing_euclidean = estate["nearest_euclidean_stt_lat"].isna() | estate["nearest_euclidean_stt_lng"].isna()
    outside_hk = (
        (estate["estate_lat"] < HK_LAT_MIN)
        | (estate["estate_lat"] > HK_LAT_MAX)
        | (estate["estate_lng"] < HK_LNG_MIN)
        | (estate["estate_lng"] > HK_LNG_MAX)
    )
    stt_lats = pd.concat(
        [estate["nearest_driving_stt_lat"], estate["nearest_euclidean_stt_lat"], stt_points["stt_lat"]],
        ignore_index=True,
    )
    stt_lngs = pd.concat(
        [estate["nearest_driving_stt_lng"], estate["nearest_euclidean_stt_lng"], stt_points["stt_lng"]],
        ignore_index=True,
    )
    return {
        "total_estates": int(len(estate)),
        "estates_with_missing_estate_coordinates": int(missing_estate.sum()),
        "estates_with_missing_google_stt_coordinates": int(missing_google.sum()),
        "estates_with_missing_euclidean_stt_coordinates": int(missing_euclidean.sum()),
        "estates_outside_hong_kong_bbox": int(outside_hk.sum()),
        "min_estate_lat": estate["estate_lat"].min(),
        "max_estate_lat": estate["estate_lat"].max(),
        "min_estate_lng": estate["estate_lng"].min(),
        "max_estate_lng": estate["estate_lng"].max(),
        "min_stt_lat": stt_lats.min(),
        "max_stt_lat": stt_lats.max(),
        "min_stt_lng": stt_lngs.min(),
        "max_stt_lng": stt_lngs.max(),
    }


def node_difference_summary(estate: pd.DataFrame) -> dict[str, Any]:
    total = int(len(estate))
    diff = int(estate["driving_node_differs_from_euclidean"].sum())
    same = total - diff
    return {
        "total_estates": total,
        "same_nearest_node_count": same,
        "different_nearest_node_count": diff,
        "different_nearest_node_share": diff / total if total else np.nan,
    }


def extent_summary(
    estate: pd.DataFrame,
    stt_points: pd.DataFrame,
    epsilon_lat: float = 0.002,
    epsilon_lng: float = 0.002,
) -> dict[str, Any]:
    lat_values = pd.concat(
        [
            estate["estate_lat"],
            estate["nearest_driving_stt_lat"],
            estate["nearest_euclidean_stt_lat"],
            stt_points["stt_lat"],
        ],
        ignore_index=True,
    )
    lng_values = pd.concat(
        [
            estate["estate_lng"],
            estate["nearest_driving_stt_lng"],
            estate["nearest_euclidean_stt_lng"],
            stt_points["stt_lng"],
        ],
        ignore_index=True,
    )
    max_data_lat = float(lat_values.max())
    min_data_lat = float(lat_values.min())
    max_data_lng = float(lng_values.max())
    min_data_lng = float(lng_values.min())
    return {
        "north_lat": max(max_data_lat, ANCHORS["north_lat"]) + epsilon_lat,
        "south_lat": min(min_data_lat, ANCHORS["south_lat"]) - epsilon_lat,
        "west_lng": min(min_data_lng, ANCHORS["west_lng"]) - epsilon_lng,
        "east_lng": max(max_data_lng, ANCHORS["east_lng"]) + epsilon_lng,
        "epsilon_lat": epsilon_lat,
        "epsilon_lng": epsilon_lng,
        "max_data_lat": max_data_lat,
        "min_data_lat": min_data_lat,
        "max_data_lng": max_data_lng,
        "min_data_lng": min_data_lng,
        "anchor_north_lat": ANCHORS["north_lat"],
        "anchor_south_lat": ANCHORS["south_lat"],
        "anchor_west_lng": ANCHORS["west_lng"],
        "anchor_east_lng": ANCHORS["east_lng"],
    }


def bbox_to_3857(extent: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox_wgs84 = gpd.GeoDataFrame(
        geometry=[box(extent["west_lng"], extent["south_lat"], extent["east_lng"], extent["north_lat"])],
        crs="EPSG:4326",
    )
    bbox_3857 = bbox_wgs84.to_crs(epsg=3857)
    xmin, ymin, xmax, ymax = bbox_3857.total_bounds
    return float(xmin), float(ymin), float(xmax), float(ymax)


def estate_gdf(estate: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        estate.copy(),
        geometry=gpd.points_from_xy(estate["estate_lng"], estate["estate_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)


def stt_nodes_gdf(stt_points: pd.DataFrame) -> gpd.GeoDataFrame:
    nodes = stt_points.copy()
    return gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes["stt_lng"], nodes["stt_lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)


def line_gdf(estate: pd.DataFrame, mode: str) -> gpd.GeoDataFrame:
    rows = []
    for _, row in estate.iterrows():
        if mode == "google":
            end_lng = row["nearest_driving_stt_lng"]
            end_lat = row["nearest_driving_stt_lat"]
            color = "#0066FF" if bool(row["driving_node_differs_from_euclidean"]) else "#B0B0B0"
        else:
            end_lng = row["nearest_euclidean_stt_lng"]
            end_lat = row["nearest_euclidean_stt_lat"]
            color = "#808080"
        if pd.isna(row["estate_lng"]) or pd.isna(row["estate_lat"]) or pd.isna(end_lng) or pd.isna(end_lat):
            continue
        rows.append(
            {
                "estate_id": row["estate_id"],
                "line_color": color,
                "geometry": LineString([(row["estate_lng"], row["estate_lat"]), (end_lng, end_lat)]),
            }
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(epsg=3857)


def color_scale(estate: pd.DataFrame) -> tuple[LinearSegmentedColormap, TwoSlopeNorm, float]:
    vmax = float(np.nanpercentile(np.abs(estate["adjusted_growth_pct"]), 95))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    cmap = LinearSegmentedColormap.from_list("red_white_green", ["#B2182B", "#FFFFFF", "#1A9850"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    return cmap, norm, vmax


def estate_marker_sizes(estate: pd.DataFrame, min_size: float = 22, max_size: float = 95) -> pd.Series:
    counts = pd.to_numeric(estate["n_total"], errors="coerce").fillna(0).clip(lower=0)
    scaled = np.sqrt(counts)
    if scaled.max() == scaled.min():
        return pd.Series((min_size + max_size) / 2, index=estate.index)
    sizes = min_size + (scaled - scaled.min()) / (scaled.max() - scaled.min()) * (max_size - min_size)
    return sizes.clip(lower=min_size, upper=max_size)


def add_basemap(ax: plt.Axes) -> None:
    cx.add_basemap(
        ax,
        source=cx.providers.CartoDB.Positron,
        attribution=False,
        reset_extent=False,
        zoom="auto",
    )


def draw_stt_nodes(ax: plt.Axes, nodes: gpd.GeoDataFrame) -> None:
    ax.scatter(
        nodes.geometry.x,
        nodes.geometry.y,
        marker="*",
        s=310,
        c=nodes["stt_color"].tolist(),
        edgecolors="black",
        linewidths=0.8,
        zorder=6,
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor=row["stt_color"],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=12,
            label=str(row["stt_name"]),
        )
        for _, row in nodes.iterrows()
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.2,
        frameon=True,
        framealpha=0.88,
        facecolor="white",
        edgecolor="0.75",
        title="STT nodes",
        title_fontsize=8,
    )
    legend.set_zorder(8)


def draw_map(
    estate: pd.DataFrame,
    stt_points: pd.DataFrame,
    extent_3857: tuple[float, float, float, float],
    mode: str,
    output_path: Path,
    ax: plt.Axes | None = None,
    add_colorbar_to_ax: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 9))
    else:
        fig = ax.figure
    xmin, ymin, xmax, ymax = extent_3857
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    add_basemap(ax)

    estates = estate_gdf(estate)
    nodes = stt_nodes_gdf(stt_points)
    lines = line_gdf(estate, mode)
    alpha = 0.48 if mode == "google" else 0.36
    lines.plot(ax=ax, color=lines["line_color"].tolist(), linewidth=0.65, alpha=alpha, zorder=2)

    cmap, norm, _ = color_scale(estate)
    estates.plot(
        ax=ax,
        column="adjusted_growth_pct",
        cmap=cmap,
        norm=norm,
        markersize=estate_marker_sizes(estate).to_numpy(),
        edgecolor="black",
        linewidth=0.25,
        alpha=0.92,
        zorder=4,
    )
    draw_stt_nodes(ax, nodes)
    ax.set_axis_off()
    if add_colorbar_to_ax:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
        cbar.set_label("Adjusted post-shock price growth (%)", fontsize=9)
    if created_fig:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    return fig, ax


def draw_side_by_side(
    estate: pd.DataFrame,
    stt_points: pd.DataFrame,
    extent_3857: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), constrained_layout=True)
    draw_map(
        estate,
        stt_points,
        extent_3857,
        "google",
        output_path,
        ax=axes[0],
        add_colorbar_to_ax=False,
    )
    draw_map(
        estate,
        stt_points,
        extent_3857,
        "euclidean",
        output_path,
        ax=axes[1],
        add_colorbar_to_ax=False,
    )
    cmap, norm, _ = color_scale(estate)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cbar.set_label("Adjusted post-shock price growth (%)", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path) -> None:
    text = """# STT Heatmap Figure Notes

## Growth measure

Adjusted post-shock price growth is measured as the post-minus-pre change in estate-level hedonic residual log price per square foot around the 8 March 2024 STT OZP gazettal. The hedonic regression controls for log area, floor group, building age, and month effects. The log change is converted to percentage terms as 100 * (exp(change) - 1).

## Google map note

Each estate is connected to the STT node with the shortest Google driving time. Blue lines indicate estates whose Google-nearest STT node differs from their Euclidean-nearest STT node. Estate color indicates adjusted post-shock price growth. Green indicates positive adjusted growth and red indicates negative adjusted growth.

## Euclidean map note

Each estate is connected to the STT node with the shortest Euclidean distance. Estate color indicates the same adjusted post-shock price growth measure.

## Caution

These maps are descriptive visualizations. They show spatial patterns in adjusted price growth, not estate-specific causal treatment effects.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    map_dir = output_dir / "maps"
    figures_dir = output_dir / "figures"
    map_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tx = load_transactions(Path(args.input).expanduser())
    stt_points = load_stt_points(Path(args.stt_points).expanduser())
    residuals = fit_hedonic_residuals(hedonic_window_sample(tx))
    write_notes(map_dir / "map_figure_notes.md")

    for threshold in args.thresholds:
        suffix = f"n{threshold}"
        estate = build_estate_growth(residuals, min_period_n=threshold)
        if estate.empty:
            raise ValueError(f"No estates remain after applying n_pre/n_post >= {threshold}")
        estate.to_csv(map_dir / f"estate_heatmap_growth_{suffix}.csv", index=False, encoding="utf-8-sig")

        write_csv(map_dir / f"map_coordinate_check_{suffix}.csv", [coordinate_check(estate, stt_points)])
        extent = extent_summary(estate, stt_points)
        write_csv(map_dir / f"map_extent_summary_{suffix}.csv", [extent])
        write_csv(
            map_dir / f"google_vs_euclidean_node_difference_summary_{suffix}.csv",
            [node_difference_summary(estate)],
        )

        extent_3857 = bbox_to_3857(extent)
        draw_map(
            estate,
            stt_points,
            extent_3857,
            "google",
            figures_dir / f"fig_google_stt_access_heatmap_{suffix}.pdf",
        )
        draw_map(
            estate,
            stt_points,
            extent_3857,
            "euclidean",
            figures_dir / f"fig_euclidean_stt_access_heatmap_{suffix}.pdf",
        )
        print(f"Done threshold {threshold}: {len(estate)} estates", flush=True)

    print("Done. Wrote map data to:", map_dir.resolve())
    print("Done. Wrote map figures to:", figures_dir.resolve())


if __name__ == "__main__":
    main()
