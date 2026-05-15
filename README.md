# San Tin Technopole Housing Price Capitalization

This repository contains a reproducible research pipeline for studying whether residential estates in northern Hong Kong responded differently to the San Tin Technopole statutory planning shock, and whether Google route-based accessibility captures the short-run capitalization gradient more clearly than Euclidean distance.

The code is organized so that each step can be rerun independently, with implementation files grouped by pipeline stage under `scripts/`.

## Research Design

The main empirical design is a transaction-level continuous-treatment DiD-style hedonic regression:

```text
log_price_psf_ijt = estate FE_i + month FE_t
                  + beta * (Access_i x Post_ijt)
                  + controls_ijt + error_ijt
```

Here, `i` indexes the estate, `j` indexes the transaction within an estate-month, and `t` indexes the calendar month. `Post_ijt` is defined from each transaction date, with the main shock dated `2024-03-08`.

Main event date:

```text
2024-03-08
```

Main event window:

```text
2023-03-08 through 2025-03-07
```

Accessibility measures:

```text
access_google = -log(nearest Google driving time to STT + 1)
access_eucl   = -log(nearest non-network distance to STT + 1)
```

The non-network benchmark is computed from coordinates using great-circle distance and is referred to as the Euclidean benchmark in the analysis. The main regressions use estate fixed effects, month fixed effects, and standard errors clustered by estate.

## Repository Structure

```text
.
├── scripts/
│   ├── data/
│   │   ├── scrape_centanet_transactions.py
│   │   └── prepare_transactions.py
│   ├── accessibility/
│   │   ├── geocode_transactions.py
│   │   ├── prepare_stt_locations.py
│   │   ├── query_stt_routes.py
│   │   ├── link_nearest_stt_by_driving.py
│   │   └── link_nearest_stt_by_euclidean.py
│   ├── analysis/
│   │   ├── run_main_regressions.py
│   │   └── run_negative_coefficient_checks.py
│   └── visualization/
│       └── run_stt_heatmap_maps.py
├── data/
│   ├── raw/
│   └── processed/
├── cache/
├── results/
│   ├── figures/
│   ├── maps/
│   ├── regression/
│   ├── report/
│   └── tables/
├── README.md
└── requirements.txt
```

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Google API calls require an environment variable:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Do not commit API keys, raw data, caches, or generated results to GitHub.

## Reproducible Pipeline

### 1. Scrape Centanet Transactions

```bash
python scripts/data/scrape_centanet_transactions.py
```

Default output:

```text
data/raw/centanet_transactions.csv
```

The crawler stores progress and field metadata under `cache/`.

### 2. Prepare Target-Area Sale Transactions

```bash
python scripts/data/prepare_transactions.py
```

Default output:

```text
data/processed/centanet_transactions_sale_target_areas.csv
```

This step keeps sale transactions, drops records without estate names, and restricts the sample to STT-relevant web scopes.

### 3. Geocode Estates

Dry run:

```bash
python scripts/accessibility/geocode_transactions.py
```

Run all missing geocodes:

```bash
python scripts/accessibility/geocode_transactions.py --run-all
```

Default outputs:

```text
cache/geocode_cache.csv
data/processed/centanet_transactions_sale_target_areas_geocoded.csv
```

The geocode query is built from estate name, parent estate name when available, admin district, and Hong Kong.

### 4. Prepare Estate Coordinates and STT Nodes

```bash
python scripts/accessibility/prepare_stt_locations.py
```

Default outputs:

```text
data/processed/estate_coordinates.csv
data/processed/stt_points.csv
```

The STT points are manually selected internal transport nodes used as accessibility anchors.

### 5. Query Google Routes

Dry run:

```bash
python scripts/accessibility/query_stt_routes.py
```

Run all missing routes:

```bash
python scripts/accessibility/query_stt_routes.py --run-all
```

Default output:

```text
cache/stt_driving_routes.csv
```

This computes estate-to-STT driving time and route distance for all estate-node pairs.

### 6. Link Nearest STT Nodes

Driving-time nearest STT:

```bash
python scripts/accessibility/link_nearest_stt_by_driving.py
```

Euclidean nearest STT:

```bash
python scripts/accessibility/link_nearest_stt_by_euclidean.py
```

Final transaction-level accessibility table:

```text
data/processed/centanet_transactions_stt_accessibility_final.csv
```

### 7. Run Main Regressions

```bash
python scripts/analysis/run_main_regressions.py
```

Main outputs:

```text
data/processed/analysis_sample_used.csv
results/regression/regression_results_main.csv
results/regression/beta_comparison_effect_size.csv
results/tables/table_main_regressions.tex
results/tables/table_beta_comparison_effect_size.tex
results/figures/fig_all_specs_coef_plot.pdf
```

The main regression sample is restricted to the ±1 year event window.

### 8. Run Diagnostic Checks

```bash
python scripts/analysis/run_negative_coefficient_checks.py
```

Main outputs:

```text
results/regression/check1_webscope_post_results.csv
results/regression/check2_first_second_control_results.csv
results/regression/check3_drop_nearest_results.csv
results/regression/check4_full_sample_results.csv
results/regression/check5_approval_shock_results.csv
results/regression/negative_coefficient_core_checks_summary.md
results/tables/check1_webscope_post_results.tex
results/tables/check2_first_second_control_results.tex
results/tables/check3_drop_nearest_results.tex
results/tables/check4_full_sample_results.tex
results/tables/check5_approval_shock_results.tex
```

The diagnostic script is separate from the main ±1 year regressions. In the current implementation, Checks 1-3 and Check 5 are estimated on the full available processed analysis sample, while Check 4 explicitly reports the full-sample version of the main specifications for comparison with the main event-window results.

### 9. Generate Maps

```bash
python scripts/visualization/run_stt_heatmap_maps.py
```

Default thresholds are `6 10 20`, meaning `n_pre >= threshold` and `n_post >= threshold`.

Main outputs:

```text
results/maps/estate_heatmap_growth_n6.csv
results/maps/estate_heatmap_growth_n10.csv
results/maps/estate_heatmap_growth_n20.csv
results/figures/fig_google_stt_access_heatmap_n6.pdf
results/figures/fig_google_stt_access_heatmap_n10.pdf
results/figures/fig_google_stt_access_heatmap_n20.pdf
results/figures/fig_euclidean_stt_access_heatmap_n6.pdf
results/figures/fig_euclidean_stt_access_heatmap_n10.pdf
results/figures/fig_euclidean_stt_access_heatmap_n20.pdf
```

The heatmap color is based on estate-level post-minus-pre hedonic residual growth, not raw price growth.

## Notes on Data and Reproducibility

- Raw Centanet data and Google API caches are excluded from Git by default.
- Generated results are excluded from Git by default.
- Reproducing the raw scrape depends on Centanet continuing to expose its rolling 3-year public transaction window.
- Downstream code can be rerun from stored inputs if the user has Google API credentials for any uncached API calls.
- Google API outputs may change slightly over time if Google updates geocoding or routing behavior.
- The scripts keep raw crawler output separate from processed analysis data.

## Main Outputs for Writing

The key LaTeX tables are:

```text
results/tables/table_main_regressions.tex
results/tables/table_beta_comparison_effect_size.tex
results/tables/check1_webscope_post_results.tex
results/tables/check2_first_second_control_results.tex
results/tables/check3_drop_nearest_results.tex
results/tables/check4_full_sample_results.tex
results/tables/check5_approval_shock_results.tex
```
