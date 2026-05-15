#!/usr/bin/env python3
"""
Run the main STT capitalization regressions.

Design:
  - Transaction-level continuous-treatment DiD-style hedonic regressions.
  - Estate fixed effects and month fixed effects in every specification.
  - Standard errors clustered by estate.
  - Four main specifications comparing Google driving-time accessibility and
    Euclidean-distance accessibility, with and without unit controls.

Primary input:
  centanet_transactions_stt_accessibility_final.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS


DEFAULT_INPUT = "data/processed/centanet_transactions_stt_accessibility_final.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_ANALYSIS_SAMPLE = "data/processed/analysis_sample_used.csv"
POST_DATE = pd.Timestamp("2024-03-08")
EVENT_WINDOW_START = pd.Timestamp("2023-03-08")
EVENT_WINDOW_END = pd.Timestamp("2025-03-07")


@dataclass
class Spec:
    name: str
    label: str
    treatment: str
    access_measure: str
    has_controls: bool


SPECS = [
    Spec("spec1_google_baseline", "Google baseline", "treat_google", "access_google", False),
    Spec("spec2_euclidean_baseline", "Euclidean baseline", "treat_eucl", "access_eucl", False),
    Spec("spec3_google_controls", "Google + controls", "treat_google", "access_google", True),
    Spec("spec4_euclidean_controls", "Euclidean + controls", "treat_eucl", "access_eucl", True),
]

REQUIRED_RAW_COLUMNS = [
    "log_price_psf",
    "geocode_query",
    "transaction_date",
    "month",
    "nearest_driving_time_min",
    "nearest_driving_route_distance_km",
    "nearest_euclidean_distance_km",
    "log_area",
    "floor_group",
    "building_age",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run main STT regression specifications.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Input CSV. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Directory for regression outputs. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--analysis-sample-output", default=DEFAULT_ANALYSIS_SAMPLE, help=f"Analysis sample CSV. Default: {DEFAULT_ANALYSIS_SAMPLE}")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))


def clean_category(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    out = series.astype("string").fillna(missing_label).str.strip()
    out = out.mask(out == "", missing_label)
    return out.astype("category")


def load_and_construct(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    require_columns(df, REQUIRED_RAW_COLUMNS)

    if "first_or_second_hand" not in df.columns:
        df["first_or_second_hand"] = "Missing"

    df["estate_id"] = df["geocode_query"].astype("string").str.strip()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"].astype("string") + "-01", errors="coerce")

    numeric_columns = [
        "log_price_psf",
        "nearest_driving_time_min",
        "nearest_driving_route_distance_km",
        "nearest_euclidean_distance_km",
        "log_area",
        "building_age",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["gtime_min"] = df["nearest_driving_time_min"]
    df["gdist_min"] = df["nearest_driving_route_distance_km"]
    df["edist_min"] = df["nearest_euclidean_distance_km"]
    df["access_google"] = -np.log(df["gtime_min"] + 1)
    df["access_eucl"] = -np.log(df["edist_min"] + 1)
    df["detour_ratio"] = df["gdist_min"] / df["edist_min"]
    df["post_ozp"] = (df["transaction_date"] >= POST_DATE).astype("Int64")
    df["treat_google"] = df["access_google"] * df["post_ozp"].astype(float)
    df["treat_eucl"] = df["access_eucl"] * df["post_ozp"].astype(float)
    df["floor_group"] = clean_category(df["floor_group"])
    df["first_or_second_hand"] = clean_category(df["first_or_second_hand"])

    base_required = [
        "log_price_psf",
        "estate_id",
        "month",
        "gtime_min",
        "gdist_min",
        "edist_min",
        "access_google",
        "access_eucl",
        "treat_google",
        "treat_eucl",
    ]
    sample = df.dropna(subset=base_required).copy()
    sample = sample.loc[sample["estate_id"].astype(str).str.len() > 0].copy()
    sample = sample.loc[
        (sample["transaction_date"] >= EVENT_WINDOW_START)
        & (sample["transaction_date"] <= EVENT_WINDOW_END)
    ].copy()
    sample["event_window_1yr"] = 1
    sample["month_id"] = sample["month"].dt.strftime("%Y-%m")
    return sample


def make_exog(df: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    exog = pd.DataFrame({spec.treatment: df[spec.treatment].astype(float)}, index=df.index)
    if spec.has_controls:
        exog["log_area"] = df["log_area"].astype(float)
        exog["building_age"] = df["building_age"].astype(float)

        floor_dummies = pd.get_dummies(
            df["floor_group"].astype("category"),
            prefix="floor_group",
            drop_first=True,
            dtype=float,
        )
        hand_dummies = pd.get_dummies(
            df["first_or_second_hand"].astype("category"),
            prefix="first_or_second_hand",
            drop_first=True,
            dtype=float,
        )
        exog = pd.concat([exog, floor_dummies, hand_dummies], axis=1)
    return exog


def spec_sample(base: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    needed = ["log_price_psf", "estate_id", "month", "month_id", spec.treatment, spec.access_measure]
    if spec.has_controls:
        needed.extend(["log_area", "building_age"])
    out = base.dropna(subset=needed).copy()
    return out


def fit_spec(base: pd.DataFrame, spec: Spec) -> tuple[Any, pd.DataFrame]:
    df = spec_sample(base, spec)
    df = df.set_index(["estate_id", "month"], drop=False)
    y = df["log_price_psf"].astype(float)
    exog = make_exog(df, spec)

    model = PanelOLS(
        y,
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return result, df


def stars(pvalue: float) -> str:
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def fmt_num(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def collect_result_row(spec: Spec, result: Any, df: pd.DataFrame) -> dict[str, Any]:
    term = spec.treatment
    beta = float(result.params[term])
    se = float(result.std_errors[term])
    pvalue = float(result.pvalues[term])
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    return {
        "specification": spec.name,
        "label": spec.label,
        "core_term": term,
        "access_measure": spec.access_measure,
        "beta": beta,
        "std_error": se,
        "p_value": pvalue,
        "ci_low_95": ci_low,
        "ci_high_95": ci_high,
        "observations": int(result.nobs),
        "estates": int(df["estate_id"].nunique()),
        "months": int(df["month_id"].nunique()),
        "estate_fe": "Yes",
        "month_fe": "Yes",
        "unit_controls": "Yes" if spec.has_controls else "No",
        "first_second_hand_control": "Yes" if spec.has_controls else "No",
        "clustered_se": "Estate",
        "r2_within": float(result.rsquared_within),
        "r2_overall": float(result.rsquared_overall),
    }


def save_single_coef_plot(row: dict[str, Any], output_dir: Path) -> None:
    beta = float(row["beta"])
    xerr = [[beta - float(row["ci_low_95"])], [float(row["ci_high_95"]) - beta]]
    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.errorbar(beta, [0], xerr=xerr, fmt="o", color="black", ecolor="black", capsize=4)
    ax.axvline(0, color="0.7", linestyle="--", linewidth=1)
    ax.set_yticks([0])
    ax.set_yticklabels([str(row["core_term"])])
    ax.set_xlabel("Coefficient value")
    ax.set_title(str(row["label"]))
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    fig.tight_layout()
    stem = output_dir / f"fig_{row['specification']}_coef"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_all_coef_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    y = np.arange(len(rows))[::-1]
    betas = np.array([float(row["beta"]) for row in rows])
    lows = np.array([float(row["ci_low_95"]) for row in rows])
    highs = np.array([float(row["ci_high_95"]) for row in rows])
    xerr = np.vstack([betas - lows, highs - betas])

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(betas, y, xerr=xerr, fmt="o", color="black", ecolor="black", capsize=4)
    ax.axvline(0, color="0.7", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient value")
    ax.set_title("Outcome: log price per square foot; SEs clustered by estate")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_all_specs_coef_plot.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"No rows to write and no fieldnames supplied: {path}")
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def fmt_latex_cell(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return latex_escape(str(value))
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def write_latex_table(
    path: Path,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    notes: str,
) -> None:
    col_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[!htbp]\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(cell) for cell in row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\textwidth}",
            r"\footnotesize",
            rf"\emph{{Notes}}: {latex_escape(notes)}",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_table_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    coef_row = ["Access $\\times$ Post"]
    se_row = [""]
    obs_row = ["Observations"]
    estates_row = ["Estates"]
    months_row = ["Months"]
    estate_fe_row = ["Estate FE"]
    month_fe_row = ["Month FE"]
    controls_row = ["Unit Controls"]
    hand_row = ["First/Second Hand Control"]
    cluster_row = ["Clustered SE"]

    for row in rows:
        coef_row.append(fmt_num(float(row["beta"]), 4) + stars(float(row["p_value"])))
        se_row.append(f"({fmt_num(float(row['std_error']), 4)})")
        obs_row.append(f"{int(row['observations']):,}")
        estates_row.append(f"{int(row['estates']):,}")
        months_row.append(f"{int(row['months']):,}")
        estate_fe_row.append(str(row["estate_fe"]))
        month_fe_row.append(str(row["month_fe"]))
        controls_row.append(str(row["unit_controls"]))
        hand_row.append(str(row["first_second_hand_control"]))
        cluster_row.append(str(row["clustered_se"]))
    return [
        coef_row,
        se_row,
        obs_row,
        estates_row,
        months_row,
        estate_fe_row,
        month_fe_row,
        controls_row,
        hand_row,
        cluster_row,
    ]


def save_regression_tables(rows: list[dict[str, Any]], tables_dir: Path, regression_dir: Path) -> None:
    columns = [
        "(1) Google baseline",
        "(2) Euclidean baseline",
        "(3) Google + controls",
        "(4) Euclidean + controls",
    ]
    table_rows = build_table_rows(rows)

    csv_rows = []
    for table_row in table_rows:
        csv_rows.append({"row": table_row[0], **{columns[i]: table_row[i + 1] for i in range(4)}})
    write_csv(regression_dir / "table_main_regressions.csv", csv_rows)

    latex_lines = [
        r"\begin{table}[!htbp]\centering",
        r"\caption{STT Accessibility and Transaction Prices}",
        r"\label{tab:main_regressions}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        " & " + " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for table_row in table_rows:
        latex_lines.append(
            latex_escape(table_row[0]).replace(r"Access \$\textbackslash{}times\$ Post", r"Access $\times$ Post")
            + " & "
            + " & ".join(latex_escape(cell) for cell in table_row[1:])
            + r" \\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\textwidth}",
            r"\footnotesize",
            r"\emph{Notes}: Outcome is log transaction price per square foot. The main sample is restricted to transactions from 2023-03-08 through 2025-03-07. All specifications include estate and month fixed effects. Standard errors are clustered by estate. Google accessibility is defined as $-\log(\text{Google driving time to nearest STT node}+1)$. Euclidean accessibility is defined as $-\log(\text{Euclidean distance to nearest STT node}+1)$. Significance levels: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (tables_dir / "table_main_regressions.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def save_regression_results_tex(rows: list[dict[str, Any]], tables_dir: Path) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["label"]),
                str(row["core_term"]),
                fmt_latex_cell(row["beta"]),
                fmt_latex_cell(row["std_error"]),
                fmt_latex_cell(row["p_value"]),
                f"{int(row['observations']):,}",
                f"{int(row['estates']):,}",
                f"{int(row['months']):,}",
            ]
        )
    write_latex_table(
        tables_dir / "regression_results_main.tex",
        "Main Regression Coefficient Summary",
        "tab:regression_results_main",
        ["Specification", "Term", "Coefficient", "SE", "p-value", "Obs.", "Estates", "Months"],
        table_rows,
        "Coefficient estimates from estate and month fixed-effects regressions. Standard errors are clustered by estate.",
    )


def effect_size_rows(result_rows: list[dict[str, Any]], samples: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for row in result_rows:
        spec_name = str(row["specification"])
        access_measure = str(row["access_measure"])
        sample = samples[spec_name]
        p25 = float(sample[access_measure].quantile(0.25))
        p75 = float(sample[access_measure].quantile(0.75))
        delta = p75 - p25
        beta = float(row["beta"])
        effect_log = beta * delta
        effect_pct = 100 * (math.exp(effect_log) - 1)
        out.append(
            {
                "specification": spec_name,
                "access_measure": access_measure,
                "beta_hat": beta,
                "p25_access": p25,
                "p75_access": p75,
                "p75_minus_p25": delta,
                "effect_log": effect_log,
                "effect_pct": effect_pct,
            }
        )
    return out


def save_effect_size(effect_rows: list[dict[str, Any]], tables_dir: Path, regression_dir: Path) -> None:
    write_csv(regression_dir / "effect_size_summary.csv", effect_rows)
    tex_rows = [
        [
            str(row["specification"]),
            str(row["access_measure"]),
            fmt_num(float(row["beta_hat"]), 4),
            fmt_num(float(row["p25_access"]), 4),
            fmt_num(float(row["p75_access"]), 4),
            fmt_num(float(row["p75_minus_p25"]), 4),
            fmt_num(float(row["effect_log"]), 4),
            fmt_num(float(row["effect_pct"]), 2),
        ]
        for row in effect_rows
    ]
    write_latex_table(
        tables_dir / "effect_size_summary.tex",
        "P25-to-P75 Accessibility Effect Sizes",
        "tab:effect_size_summary",
        ["Specification", "Access", "Beta", "P25", "P75", "P75-P25", "Log effect", "Percent effect"],
        tex_rows,
        "Effects are calculated as beta times the interquartile range of the relevant accessibility measure. Percent effects equal 100 times exp(log effect) minus 1.",
    )


def beta_comparison_rows(result_rows: list[dict[str, Any]], samples: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    for row in result_rows:
        spec_name = str(row["specification"])
        access_measure = str(row["access_measure"])
        sample = samples[spec_name]
        access = pd.to_numeric(sample[access_measure], errors="coerce").dropna()
        p25 = float(access.quantile(0.25))
        p75 = float(access.quantile(0.75))
        iqr = p75 - p25
        sd = float(access.std(ddof=1))
        beta = float(row["beta"])
        iqr_effect_log = beta * iqr
        standardized_beta = beta * sd
        out.append(
            {
                "spec": str(row["label"]),
                "access_measure": access_measure,
                "raw_beta": beta,
                "standard_error": float(row["std_error"]),
                "p_value": float(row["p_value"]),
                "p25_access": p25,
                "p75_access": p75,
                "iqr_access": iqr,
                "sd_access": sd,
                "iqr_effect_log": iqr_effect_log,
                "iqr_effect_pct": 100 * (math.exp(iqr_effect_log) - 1),
                "standardized_beta": standardized_beta,
                "standardized_effect_pct": 100 * (math.exp(standardized_beta) - 1),
            }
        )
    return out


def save_beta_comparison_table(rows: list[dict[str, Any]], tables_dir: Path, regression_dir: Path) -> None:
    write_csv(regression_dir / "beta_comparison_effect_size.csv", rows)
    columns = [str(row["spec"]) for row in rows]
    table_rows = [
        ["Raw beta", *[fmt_num(float(row["raw_beta"]), 4) for row in rows]],
        ["Standard error", *[fmt_num(float(row["standard_error"]), 4) for row in rows]],
        ["p-value", *[fmt_num(float(row["p_value"]), 3) for row in rows]],
        ["Access p25", *[fmt_num(float(row["p25_access"]), 4) for row in rows]],
        ["Access p75", *[fmt_num(float(row["p75_access"]), 4) for row in rows]],
        ["Access IQR", *[fmt_num(float(row["iqr_access"]), 4) for row in rows]],
        ["IQR effect, log points", *[fmt_num(float(row["iqr_effect_log"]), 4) for row in rows]],
        ["IQR effect, %", *[fmt_num(float(row["iqr_effect_pct"]), 2) for row in rows]],
        ["Standardized beta", *[fmt_num(float(row["standardized_beta"]), 4) for row in rows]],
        ["Standardized effect, %", *[fmt_num(float(row["standardized_effect_pct"]), 2) for row in rows]],
    ]
    lines = [
        r"\begin{table}[!htbp]\centering",
        r"\caption{Comparable Effect Sizes of STT Accessibility Measures}",
        r"\label{tab:beta_effect_size_comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        " & " + " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for table_row in table_rows:
        lines.append(latex_escape(table_row[0]) + " & " + " & ".join(latex_escape(cell) for cell in table_row[1:]) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\textwidth}",
            r"\footnotesize",
            r"\emph{Notes}: This table converts the main regression coefficients into comparable effect sizes. Raw Google and Euclidean coefficients are not directly comparable because the underlying accessibility variables are measured on different scales. IQR effects are computed as $\hat{\beta}\times(P75-P25)$ of the corresponding accessibility measure and converted to percentages using $100[\exp(\cdot)-1]$. Standardized beta is computed as $\hat{\beta}\times SD(\text{access})$, leaving the outcome in log price per square foot. The regression specifications correspond to the four columns in Table \ref{tab:main_regressions}.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    (tables_dir / "table_beta_comparison_effect_size.tex").write_text("\n".join(lines), encoding="utf-8")


def save_beta_comparison_summary(rows: list[dict[str, Any]], regression_dir: Path) -> None:
    by_spec = {str(row["spec"]): row for row in rows}
    google_base = by_spec["Google baseline"]
    eucl_base = by_spec["Euclidean baseline"]
    google_controls = by_spec["Google + controls"]
    eucl_controls = by_spec["Euclidean + controls"]

    def effect_text(row: dict[str, Any]) -> str:
        return f"{float(row['iqr_effect_log']):.4f} log points ({float(row['iqr_effect_pct']):.2f}%)"

    def std_text(row: dict[str, Any]) -> str:
        return f"{float(row['standardized_beta']):.4f} log points ({float(row['standardized_effect_pct']):.2f}%)"

    base_larger = "Google" if abs(float(google_base["iqr_effect_log"])) > abs(float(eucl_base["iqr_effect_log"])) else "Euclidean"
    controls_larger = "Google" if abs(float(google_controls["iqr_effect_log"])) > abs(float(eucl_controls["iqr_effect_log"])) else "Euclidean"
    base_std_larger = "Google" if abs(float(google_base["standardized_beta"])) > abs(float(eucl_base["standardized_beta"])) else "Euclidean"
    controls_std_larger = "Google" if abs(float(google_controls["standardized_beta"])) > abs(float(eucl_controls["standardized_beta"])) else "Euclidean"

    text = f"""# Comparable Effect Sizes Summary

Raw beta values should not be directly compared because Google and Euclidean accessibility are measured on different scales.

The IQR effect is a more comparable measure of economic magnitude. In the baseline specifications, the Google IQR effect is {effect_text(google_base)}, while the Euclidean IQR effect is {effect_text(eucl_base)}; after rescaling by the interquartile range, the {base_larger} specification implies the larger absolute post-announcement effect. With controls, the Google IQR effect is {effect_text(google_controls)}, while the Euclidean IQR effect is {effect_text(eucl_controls)}; the {controls_larger} specification implies the larger absolute effect.

The standardized beta compares a one-standard-deviation change in each accessibility measure. In the baseline specifications, the Google standardized effect is {std_text(google_base)}, while the Euclidean standardized effect is {std_text(eucl_base)}; the {base_std_larger} specification is larger in absolute value. With controls, the Google standardized effect is {std_text(google_controls)}, while the Euclidean standardized effect is {std_text(eucl_controls)}; the {controls_std_larger} specification is larger in absolute value.

This table is not a new regression. It is a scale-adjusted interpretation of the four main regression coefficients.
"""
    (regression_dir / "beta_comparison_effect_size_summary.md").write_text(text, encoding="utf-8")


def save_interpretation(output_dir: Path) -> None:
    text = """# Main Regression Interpretation

- A positive coefficient on `treat_google` means that after the STT announcement, estates with shorter Google driving time to STT experienced larger relative increases in log price per square foot.
- A positive coefficient on `treat_eucl` means that after the STT announcement, estates with shorter straight-line distance to STT experienced larger relative increases in log price per square foot.
- The key comparison is whether the Google specifications show a stronger, more precise, or more economically meaningful coefficient than the Euclidean specifications.
- Interpret results as DiD-style pilot evidence, not definitive causal proof.
"""
    (output_dir / "regression_interpretation.md").write_text(text, encoding="utf-8")


def clean_stale_outputs(figures_dir: Path, regression_dir: Path) -> None:
    for path in figures_dir.glob("*.png"):
        path.unlink()
    for name in ["effect_size_summary.md", "table_main_regressions.md"]:
        path = regression_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    regression_dir = output_dir / "regression"
    for directory in (figures_dir, tables_dir, regression_dir):
        directory.mkdir(parents=True, exist_ok=True)
    clean_stale_outputs(figures_dir, regression_dir)

    sample = load_and_construct(Path(args.input).expanduser())
    analysis_sample_path = Path(args.analysis_sample_output).expanduser()
    analysis_sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(analysis_sample_path, index=False, encoding="utf-8-sig")

    result_rows: list[dict[str, Any]] = []
    samples: dict[str, pd.DataFrame] = {}
    for spec in SPECS:
        print(f"Running {spec.name}...", flush=True)
        result, df_spec = fit_spec(sample, spec)
        row = collect_result_row(spec, result, df_spec)
        result_rows.append(row)
        samples[spec.name] = df_spec.reset_index(drop=True)
        save_single_coef_plot(row, figures_dir)

    write_csv(regression_dir / "regression_results_main.csv", result_rows)
    save_regression_results_tex(result_rows, tables_dir)
    save_all_coef_plot(result_rows, figures_dir)
    save_regression_tables(result_rows, tables_dir, regression_dir)
    effects = effect_size_rows(result_rows, samples)
    save_effect_size(effects, tables_dir, regression_dir)
    beta_comparison = beta_comparison_rows(result_rows, samples)
    save_beta_comparison_table(beta_comparison, tables_dir, regression_dir)
    save_beta_comparison_summary(beta_comparison, regression_dir)
    save_interpretation(regression_dir)

    print("Done. Wrote figures to:", figures_dir.resolve())
    print("Done. Wrote tables to:", tables_dir.resolve())
    print("Done. Wrote regression outputs to:", regression_dir.resolve())
    print("Done. Wrote analysis sample to:", analysis_sample_path.resolve())


if __name__ == "__main__":
    main()
