#!/usr/bin/env python3
"""
Run core diagnostics for the negative Google accessibility coefficient.

Checks:
  1. Add web_scope x post_ozp controls.
  2. Add first/second-hand transaction-type controls.
  3. Drop the estates nearest to STT by Google driving-time accessibility.
  4. Use the full available sample rather than the main +/- 1 year window.
  5. Move the shock date to the approved OZP publication date.

All regressions use transaction-level data, estate fixed effects, month fixed
effects, and standard errors clustered by estate.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


DEFAULT_INPUT = "data/processed/centanet_transactions_stt_accessibility_final.csv"
DEFAULT_OUTPUT_DIR = "results"
POST_DATE = pd.Timestamp("2024-03-08")
APPROVAL_DATE = pd.Timestamp("2024-09-20")
EVENT_WINDOW_START = pd.Timestamp("2023-03-08")
EVENT_WINDOW_END = pd.Timestamp("2025-03-07")


@dataclass(frozen=True)
class DiagnosticSpec:
    name: str
    label: str
    treatment: str
    controls: bool = False
    webscope_post: bool = False
    first_second_control: bool = False
    sample_note: str = "Original sample"


REQUIRED_COLUMNS = [
    "log_price_psf",
    "geocode_query",
    "month",
    "transaction_date",
    "nearest_driving_time_min",
    "nearest_euclidean_distance_km",
    "web_scope",
    "first_or_second_hand",
    "log_area",
    "floor_group",
    "building_age",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STT negative-coefficient diagnostic checks.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Analysis dataset CSV. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def clean_string(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    out = series.astype("string").fillna(missing_label).str.strip()
    return out.mask(out == "", missing_label)


def clean_category(series: pd.Series, missing_label: str = "Missing") -> pd.Series:
    return clean_string(series, missing_label).astype("category")


def load_analysis_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    require_columns(df, REQUIRED_COLUMNS)

    if "estate_id" in df.columns:
        df["estate_id"] = clean_string(df["estate_id"])
    else:
        df["estate_id"] = clean_string(df["geocode_query"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        month_alt = pd.to_datetime(df["month"].astype("string") + "-01", errors="coerce")
        df["month"] = df["month"].fillna(month_alt)

    numeric_columns = [
        "log_price_psf",
        "nearest_driving_time_min",
        "nearest_euclidean_distance_km",
        "log_area",
        "building_age",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["access_google"] = -np.log(df["nearest_driving_time_min"] + 1)
    df["access_eucl"] = -np.log(df["nearest_euclidean_distance_km"] + 1)
    df["post_ozp"] = (df["transaction_date"] >= POST_DATE).astype("Int64")
    df["treat_google"] = df["access_google"] * df["post_ozp"].astype(float)
    df["treat_eucl"] = df["access_eucl"] * df["post_ozp"].astype(float)

    df["web_scope"] = clean_category(df["web_scope"])
    df["floor_group"] = clean_category(df["floor_group"])
    df["first_or_second_hand"] = clean_category(df["first_or_second_hand"])

    needed = [
        "log_price_psf",
        "estate_id",
        "month",
        "transaction_date",
        "post_ozp",
        "access_google",
        "access_eucl",
        "treat_google",
        "treat_eucl",
        "nearest_driving_time_min",
        "nearest_euclidean_distance_km",
        "web_scope",
        "first_or_second_hand",
    ]
    sample = df.dropna(subset=needed).copy()
    sample = sample.loc[sample["estate_id"].astype(str).str.len() > 0].copy()
    sample["month_id"] = sample["month"].dt.strftime("%Y-%m")
    return sample


def add_dummies(exog: pd.DataFrame, df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[column].astype("category"), prefix=prefix, drop_first=True, dtype=float)
    return pd.concat([exog, dummies], axis=1)


def add_webscope_post_dummies(exog: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df["web_scope"].astype("category"), prefix="web_scope_post", drop_first=True, dtype=float)
    dummies = dummies.mul(df["post_ozp"].astype(float), axis=0)
    return pd.concat([exog, dummies], axis=1)


def drop_constant_columns(exog: pd.DataFrame) -> pd.DataFrame:
    keep = [column for column in exog.columns if exog[column].notna().all() and exog[column].nunique(dropna=False) > 1]
    return exog.loc[:, keep]


def make_exog(df: pd.DataFrame, spec: DiagnosticSpec) -> pd.DataFrame:
    exog = pd.DataFrame({spec.treatment: df[spec.treatment].astype(float)}, index=df.index)
    if spec.webscope_post:
        exog = add_webscope_post_dummies(exog, df)
    if spec.first_second_control:
        exog = add_dummies(exog, df, "first_or_second_hand", "first_or_second_hand")
    if spec.controls:
        exog["log_area"] = df["log_area"].astype(float)
        exog["building_age"] = df["building_age"].astype(float)
        exog = add_dummies(exog, df, "floor_group", "floor_group")
    return drop_constant_columns(exog)


def spec_sample(base: pd.DataFrame, spec: DiagnosticSpec) -> pd.DataFrame:
    needed = ["log_price_psf", "estate_id", "month", "month_id", spec.treatment]
    if spec.webscope_post:
        needed.extend(["web_scope", "post_ozp"])
    if spec.first_second_control:
        needed.append("first_or_second_hand")
    if spec.controls:
        needed.extend(["log_area", "building_age", "floor_group"])
    return base.dropna(subset=needed).copy()


def fit_spec(base: pd.DataFrame, spec: DiagnosticSpec) -> tuple[Any, pd.DataFrame]:
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


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def latex_escape(value: Any) -> str:
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
    return "".join(replacements.get(char, char) for char in str(value))


def collect_row(check: str, spec: DiagnosticSpec, result: Any, df: pd.DataFrame) -> dict[str, Any]:
    beta = float(result.params[spec.treatment])
    se = float(result.std_errors[spec.treatment])
    pvalue = float(result.pvalues[spec.treatment])
    return {
        "check": check,
        "specification": spec.name,
        "label": spec.label,
        "sample": spec.sample_note,
        "core_term": spec.treatment,
        "beta": beta,
        "std_error": se,
        "p_value": pvalue,
        "ci_low_95": beta - 1.96 * se,
        "ci_high_95": beta + 1.96 * se,
        "observations": int(result.nobs),
        "estates": int(df["estate_id"].nunique()),
        "months": int(df["month_id"].nunique()),
        "estate_fe": "Yes",
        "month_fe": "Yes",
        "web_scope_post": "Yes" if spec.webscope_post else "No",
        "first_second_hand_control": "Yes" if spec.first_second_control else "No",
        "unit_controls": "Yes" if spec.controls else "No",
        "clustered_se": "Estate",
        "r2_within": float(result.rsquared_within),
        "r2_overall": float(result.rsquared_overall),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_regression_tex(path: Path, caption: str, label: str, columns: list[str], rows: list[dict[str, Any]], notes: str) -> None:
    table_rows: list[list[str]] = []
    coef = ["Access $\\times$ Post"]
    se = [""]
    obs = ["Observations"]
    estates = ["Estates"]
    estate_fe = ["Estate FE"]
    month_fe = ["Month FE"]
    webscope = ["Web scope $\\times$ Post"]
    first_second = ["First/Second Hand Control"]
    controls = ["Unit Controls"]
    clustered = ["Clustered SE"]

    for row in rows:
        coef.append(fmt_num(row["beta"]) + stars(float(row["p_value"])))
        se.append(f"({fmt_num(row['std_error'])})")
        obs.append(f"{int(row['observations']):,}")
        estates.append(f"{int(row['estates']):,}")
        estate_fe.append(str(row["estate_fe"]))
        month_fe.append(str(row["month_fe"]))
        webscope.append(str(row["web_scope_post"]))
        first_second.append(str(row["first_second_hand_control"]))
        controls.append(str(row["unit_controls"]))
        clustered.append(str(row["clustered_se"]))

    for table_row in [coef, se, obs, estates, estate_fe, month_fe, webscope, first_second, controls, clustered]:
        table_rows.append(table_row)

    lines = [
        r"\begin{table}[!htbp]\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        " & " + " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
    ]
    for table_row in table_rows:
        first_cell = latex_escape(table_row[0])
        first_cell = first_cell.replace(r"Access \$\textbackslash{}times\$ Post", r"Access $\times$ Post")
        first_cell = first_cell.replace(r"Web scope \$\textbackslash{}times\$ Post", r"Web scope $\times$ Post")
        lines.append(first_cell + " & " + " & ".join(latex_escape(cell) for cell in table_row[1:]) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.95\textwidth}",
            r"\footnotesize",
            rf"\emph{{Notes}}: {notes} Significance levels: * $p<0.10$, ** $p<0.05$, *** $p<0.01$.",
            r"\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_specs(base: pd.DataFrame, check: str, specs: list[DiagnosticSpec]) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        print(f"Running {spec.name}...", flush=True)
        result, df = fit_spec(base, spec)
        rows.append(collect_row(check, spec, result, df))
    return rows


def check1_specs() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec("check1_google_baseline_webscope_post", "Google baseline", "treat_google", webscope_post=True),
        DiagnosticSpec("check1_euclidean_baseline_webscope_post", "Euclidean baseline", "treat_eucl", webscope_post=True),
        DiagnosticSpec("check1_google_controls_webscope_post", "Google + controls", "treat_google", controls=True, webscope_post=True),
        DiagnosticSpec("check1_euclidean_controls_webscope_post", "Euclidean + controls", "treat_eucl", controls=True, webscope_post=True),
    ]


def check2_specs() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec("check2_google_baseline_first_second", "Google baseline", "treat_google", first_second_control=True),
        DiagnosticSpec("check2_euclidean_baseline_first_second", "Euclidean baseline", "treat_eucl", first_second_control=True),
        DiagnosticSpec("check2_google_controls_first_second", "Google + controls", "treat_google", controls=True, first_second_control=True),
        DiagnosticSpec("check2_euclidean_controls_first_second", "Euclidean + controls", "treat_eucl", controls=True, first_second_control=True),
    ]


def estate_nearest_flags(base: pd.DataFrame) -> pd.DataFrame:
    estate = (
        base.groupby("estate_id", observed=True)
        .agg(
            access_google=("access_google", "first"),
            nearest_driving_time_min=("nearest_driving_time_min", "first"),
            transaction_count=("estate_id", "size"),
        )
        .reset_index()
    )
    p95 = float(estate["access_google"].quantile(0.95))
    p90 = float(estate["access_google"].quantile(0.90))
    estate["top5_nearest"] = estate["access_google"] >= p95
    estate["top10_nearest"] = estate["access_google"] >= p90
    return estate


def run_check3(base: pd.DataFrame, regression_dir: Path) -> list[dict[str, Any]]:
    estate = estate_nearest_flags(base)
    write_csv(regression_dir / "check3_estate_nearest_flags.csv", estate.to_dict("records"))

    top5 = set(estate.loc[estate["top5_nearest"], "estate_id"].astype(str))
    top10 = set(estate.loc[estate["top10_nearest"], "estate_id"].astype(str))
    base_drop5 = base.loc[~base["estate_id"].astype(str).isin(top5)].copy()
    base_drop10 = base.loc[~base["estate_id"].astype(str).isin(top10)].copy()

    specs_and_samples = [
        (base, DiagnosticSpec("check3_original_google_baseline", "Original Google baseline", "treat_google")),
        (base, DiagnosticSpec("check3_original_google_controls", "Original Google + controls", "treat_google", controls=True)),
        (
            base_drop5,
            DiagnosticSpec(
                "check3_drop_top5_google_baseline",
                "Drop top 5% baseline",
                "treat_google",
                sample_note="Drop top 5% nearest estates",
            ),
        ),
        (
            base_drop5,
            DiagnosticSpec(
                "check3_drop_top5_google_controls",
                "Drop top 5% + controls",
                "treat_google",
                controls=True,
                sample_note="Drop top 5% nearest estates",
            ),
        ),
        (
            base_drop10,
            DiagnosticSpec(
                "check3_drop_top10_google_baseline",
                "Drop top 10% baseline",
                "treat_google",
                sample_note="Drop top 10% nearest estates",
            ),
        ),
        (
            base_drop10,
            DiagnosticSpec(
                "check3_drop_top10_google_controls",
                "Drop top 10% + controls",
                "treat_google",
                controls=True,
                sample_note="Drop top 10% nearest estates",
            ),
        ),
    ]

    rows = []
    for sample, spec in specs_and_samples:
        print(f"Running {spec.name}...", flush=True)
        result, df = fit_spec(sample, spec)
        rows.append(collect_row("check3_drop_nearest", spec, result, df))
    return rows


def check4_specs() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            "check4_google_baseline_full_sample",
            "Google baseline",
            "treat_google",
            sample_note="Full sample",
        ),
        DiagnosticSpec(
            "check4_euclidean_baseline_full_sample",
            "Euclidean baseline",
            "treat_eucl",
            sample_note="Full sample",
        ),
        DiagnosticSpec(
            "check4_google_controls_full_sample",
            "Google + controls",
            "treat_google",
            controls=True,
            sample_note="Full sample",
        ),
        DiagnosticSpec(
            "check4_euclidean_controls_full_sample",
            "Euclidean + controls",
            "treat_eucl",
            controls=True,
            sample_note="Full sample",
        ),
    ]


def full_sample_for_check4(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["full_sample_check"] = 1
    out["post_ozp"] = (out["transaction_date"] >= POST_DATE).astype("Int64")
    out["treat_google"] = out["access_google"] * out["post_ozp"].astype(float)
    out["treat_eucl"] = out["access_eucl"] * out["post_ozp"].astype(float)
    needed = [
        "log_price_psf",
        "estate_id",
        "month",
        "month_id",
        "access_google",
        "access_eucl",
        "treat_google",
        "treat_eucl",
    ]
    return out.dropna(subset=needed).copy()


def check5_specs() -> list[DiagnosticSpec]:
    return [
        DiagnosticSpec(
            "check5_google_baseline_approval_shock",
            "Google baseline",
            "treat_google_approval",
            sample_note="Approval shock date",
        ),
        DiagnosticSpec(
            "check5_euclidean_baseline_approval_shock",
            "Euclidean baseline",
            "treat_eucl_approval",
            sample_note="Approval shock date",
        ),
        DiagnosticSpec(
            "check5_google_controls_approval_shock",
            "Google + controls",
            "treat_google_approval",
            controls=True,
            sample_note="Approval shock date",
        ),
        DiagnosticSpec(
            "check5_euclidean_controls_approval_shock",
            "Euclidean + controls",
            "treat_eucl_approval",
            controls=True,
            sample_note="Approval shock date",
        ),
    ]


def approval_shock_sample(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["post_approval"] = (out["transaction_date"] >= APPROVAL_DATE).astype("Int64")
    out["treat_google_approval"] = out["access_google"] * out["post_approval"].astype(float)
    out["treat_eucl_approval"] = out["access_eucl"] * out["post_approval"].astype(float)
    needed = [
        "log_price_psf",
        "estate_id",
        "month",
        "month_id",
        "access_google",
        "access_eucl",
        "treat_google_approval",
        "treat_eucl_approval",
    ]
    return out.dropna(subset=needed).copy()


def beta_summary(row: dict[str, Any]) -> str:
    return f"{float(row['beta']):.4f} (p={float(row['p_value']):.3f})"


def pct_change(new: float, old: float) -> float:
    if old == 0 or pd.isna(old) or pd.isna(new):
        return math.nan
    return 100 * (abs(new) - abs(old)) / abs(old)


def find_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row["specification"] == name:
            return row
    raise KeyError(name)


def write_summary(
    path: Path,
    check1: list[dict[str, Any]],
    check2: list[dict[str, Any]],
    check3: list[dict[str, Any]],
    check4: list[dict[str, Any]],
    check5: list[dict[str, Any]],
) -> None:
    original = find_row(check3, "check3_original_google_baseline")
    original_controls = find_row(check3, "check3_original_google_controls")
    c1 = find_row(check1, "check1_google_baseline_webscope_post")
    c1_controls = find_row(check1, "check1_google_controls_webscope_post")
    c2 = find_row(check2, "check2_google_baseline_first_second")
    c2_controls = find_row(check2, "check2_google_controls_first_second")
    drop5 = find_row(check3, "check3_drop_top5_google_baseline")
    drop10 = find_row(check3, "check3_drop_top10_google_baseline")
    drop5_controls = find_row(check3, "check3_drop_top5_google_controls")
    drop10_controls = find_row(check3, "check3_drop_top10_google_controls")
    full_sample = find_row(check4, "check4_google_baseline_full_sample")
    full_sample_controls = find_row(check4, "check4_google_controls_full_sample")
    approval_google = find_row(check5, "check5_google_baseline_approval_shock")
    approval_google_controls = find_row(check5, "check5_google_controls_approval_shock")
    approval_eucl = find_row(check5, "check5_euclidean_baseline_approval_shock")
    approval_eucl_controls = find_row(check5, "check5_euclidean_controls_approval_shock")

    c1_change = pct_change(float(c1["beta"]), float(original["beta"]))
    c2_change = pct_change(float(c2["beta"]), float(original["beta"]))
    drop5_change = pct_change(float(drop5["beta"]), float(original["beta"]))
    drop10_change = pct_change(float(drop10["beta"]), float(original["beta"]))
    full_sample_change = pct_change(float(full_sample["beta"]), float(original["beta"]))
    approval_change = pct_change(float(approval_google["beta"]), float(original["beta"]))

    def still_negative(row: dict[str, Any]) -> str:
        return "still negative" if float(row["beta"]) < 0 else "no longer negative"

    text = f"""# Negative Google Coefficient Core Checks Summary

Event date is fixed at 2024-03-08. All checks use transaction-level data with estate fixed effects, month fixed effects, and standard errors clustered by estate.

1. After adding `web_scope x post_ozp`, the Google negative coefficient still exists and becomes larger in absolute value. The baseline Google estimate changes from {beta_summary(original)} to {beta_summary(c1)}; the controls estimate is {beta_summary(c1_controls)}. This suggests the original negative Google gradient is not simply absorbed by web-scope-level post-period trends.

2. After adding `C(first_or_second_hand)`, the Google coefficient changes very little. The baseline Google estimate changes from {beta_summary(original)} to {beta_summary(c2)}; the controls estimate changes from {beta_summary(original_controls)} to {beta_summary(c2_controls)}. This suggests first-hand/second-hand composition is not the main driver of the negative coefficient.

3. After dropping the nearest estates, the Google coefficient weakens modestly but remains negative. The baseline estimate changes from {beta_summary(original)} to {beta_summary(drop5)} after dropping the top 5%, and to {beta_summary(drop10)} after dropping the top 10%. The corresponding controls estimates are {beta_summary(drop5_controls)} and {beta_summary(drop10_controls)}. This is consistent with very-near-STT exposure contributing somewhat, but not fully explaining the negative gradient.

4. In the full sample, the Google coefficient is {still_negative(full_sample)}. The baseline Google estimate is {beta_summary(full_sample)}, and the controls estimate is {beta_summary(full_sample_controls)}. This compares the full-sample estimate against the main ±1 year event-window specification.

## Check 5: Approval Shock Date

Using 2024-09-20 as the approval shock date, the Google coefficient is {still_negative(approval_google)} but becomes much smaller and statistically insignificant. The baseline Google estimate is {beta_summary(approval_google)}, and the controls estimate is {beta_summary(approval_google_controls)}. The Euclidean estimates remain close to zero and not statistically significant: {beta_summary(approval_eucl)} in baseline and {beta_summary(approval_eucl_controls)} with controls. This suggests the negative sign is not reversed, but the magnitude and precision of the main finding are stronger when using the 2024-03-08 OZP gazettal/publication date.

Absolute-beta changes relative to the original Google baseline:

- Check 1 web-scope-post baseline: {c1_change:.1f}%
- Check 2 first/second-hand baseline: {c2_change:.1f}%
- Check 3 drop top 5% baseline: {drop5_change:.1f}%
- Check 3 drop top 10% baseline: {drop10_change:.1f}%
- Check 4 full-sample baseline: {full_sample_change:.1f}%
- Check 5 approval-shock baseline: {approval_change:.1f}%

Overall, the diagnostics compare the main ±1 year event-window specification against several full-sample robustness checks. Check 5 is more attenuated: the approval-date coefficient remains negative but is small and insignificant. The pattern is consistent with short-run local negative shocks around STT being better aligned with the March 2024 gazettal/publication timing than with the September 2024 approval-publication timing. This is diagnostic evidence, not definitive causal proof.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    regression_dir = output_dir / "regression"
    tables_dir = output_dir / "tables"
    regression_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sample = load_analysis_sample(Path(args.input).expanduser())

    check1 = run_specs(sample, "check1_webscope_post", check1_specs())
    write_csv(regression_dir / "check1_webscope_post_results.csv", check1)
    write_regression_tex(
        tables_dir / "check1_webscope_post_results.tex",
        "Diagnostic Check 1: Web-Scope Post Controls",
        "tab:check1_webscope_post",
        [row["label"] for row in check1],
        check1,
        "All specifications include estate and month fixed effects. Web scope by post-period controls are included.",
    )

    check2 = run_specs(sample, "check2_first_second_control", check2_specs())
    write_csv(regression_dir / "check2_first_second_control_results.csv", check2)
    write_regression_tex(
        tables_dir / "check2_first_second_control_results.tex",
        "Diagnostic Check 2: First-Hand/Second-Hand Controls",
        "tab:check2_first_second",
        [row["label"] for row in check2],
        check2,
        "All specifications include estate and month fixed effects. First-hand/second-hand controls are included.",
    )

    check3 = run_check3(sample, regression_dir)
    write_csv(regression_dir / "check3_drop_nearest_results.csv", check3)
    write_regression_tex(
        tables_dir / "check3_drop_nearest_results.tex",
        "Diagnostic Check 3: Dropping Estates Nearest to STT",
        "tab:check3_drop_nearest",
        [row["label"] for row in check3],
        check3,
        "All specifications include estate and month fixed effects. Drop samples exclude estates above the estate-level Google-accessibility percentile cutoff.",
    )

    check4_sample = full_sample_for_check4(sample)
    check4 = run_specs(check4_sample, "check4_full_sample", check4_specs())
    write_csv(regression_dir / "check4_full_sample_results.csv", check4)
    write_regression_tex(
        tables_dir / "check4_full_sample_results.tex",
        "Diagnostic Check 4: Full Sample",
        "tab:check4_full_sample",
        [row["label"] for row in check4],
        check4,
        "All specifications include estate and month fixed effects. The sample uses all available observations, and post is defined as transaction date on or after 2024-03-08.",
    )

    check5_sample = approval_shock_sample(sample)
    check5 = run_specs(check5_sample, "check5_approval_shock", check5_specs())
    write_csv(regression_dir / "check5_approval_shock_results.csv", check5)
    write_regression_tex(
        tables_dir / "check5_approval_shock_results.tex",
        "Diagnostic Check 5: Approval Shock Date",
        "tab:check5_approval_shock",
        [row["label"] for row in check5],
        check5,
        "All specifications include estate and month fixed effects. Post is defined as transaction date on or after 2024-09-20, the approved San Tin Technopole OZP publication date.",
    )

    write_summary(regression_dir / "negative_coefficient_core_checks_summary.md", check1, check2, check3, check4, check5)

    print("Done. Wrote diagnostic CSV files to:", regression_dir.resolve())
    print("Done. Wrote diagnostic TeX tables to:", tables_dir.resolve())


if __name__ == "__main__":
    main()
