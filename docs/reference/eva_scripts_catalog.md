# Eva Scripts Catalog

All evaluation scripts in [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) and important utility scripts in [`scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts).

---

## Prerequisites

Most evaluation scripts require time series data in `_TS/`. These files are **auto-generated** by the evaluation scripts themselves if they are missing. However, the following prerequisite scripts should be run first:

```bash
# Convert y_pred CSVs to parquet format
python scripts/convert_y_pred_to_parquet.py --EXP_TITLE full_exp_jan

# Compute dataset-dependent random baseline slopes
python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE full_exp_jan
```

Some scripts also need leaderboard rankings. Generate them with:

```bash
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
```

---

## 5 Canonical Recipes

The scripts most researchers need:

### 1. Generate Leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

### 2. Generate Example Learning Curve Plot

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

**Output:** `plots/single_learning_curve/*.parquet`, `plots/single_learning_curve.pdf`

!!! note
    This script generates an **example learning curve plot** for illustration purposes. It also auto-generates `_TS/*.parquet` if missing, but so do most other evaluation scripts.

### 3. Compute Three Correlations (Paper Heatmaps)

```bash
# Metric-based (Pearson) → Blue heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan

# Queried samples (Jaccard) → Green heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan

# Ranking invariance (Kendall) → Orange heatmaps
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

### 4. Runtime Analysis

```bash
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

**Output:** `plots/runtime/query_selection_time.parquet`

### 5. Publication-Ready Plots (all figures)

```bash
python -m eva_scripts.redo_plots_for_paper --EXP_TITLE full_exp_jan
```

---

## Complete Eva Scripts Index

### Core Analysis Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `learning_curve.py` | Per-cycle CSVs, `05_done_workload.csv` | `plots/single_learning_curve/*.parquet`, PDF | Generates an example learning curve plot for illustration. Auto-generates `_TS/*.parquet` if missing (as do most other eva_scripts). |
| `calculate_leaderboard_rankings.py` | `_TS/*.parquet` | Ranking parquets (multiple interpolation modes) | Generates strategy rankings across datasets using different metrics and interpolation methods. |
| `final_leaderboard.py` | `_TS/*.parquet`, ranking data | `plots/final_leaderboard/*.parquet` | Main leaderboard generation — ranks strategies and produces the paper's Table 1. |
| `runtime.py` | `query_selection_time.csv.xz` | `plots/runtime/query_selection_time.parquet` | Analyzes and plots query selection time distributions per strategy. |

### Correlation & Hyperparameter Analysis Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `basic_metrics_correlation.py` | Per-cycle CSVs (accuracy, F1, etc.) | `plots/basic_metrics/Standard Metrics.parquet` | Pearson correlation matrix between standard ML metrics. |
| `auc_metric_correlation.py` | AUC metric parquets | `plots/AUC/auc_*.parquet` | Pearson correlation between AUC-based aggregation metrics. |
| `single_hyperparameter_evaluation_metric.py` | `_TS/*.parquet` | `plots/single_hyperparameter/*/` (Blue heatmaps) | Metric-based (Pearson) correlation — how metric outcomes change when varying one hyperparameter. |
| `single_hyperparameter_evaluation_indices.py` | `selected_indices.parquet` | `plots/single_hyperparameter/*/` (Green heatmaps) | Jaccard similarity of queried samples — do strategies select the same samples under different hyperparameters? |
| `leaderboard_single_hyperparameter_influence.py` | `_TS/*.parquet` | Single hyperparameter influence parquets | Kendall τ ranking invariance — how much does changing one hyperparameter affect strategy ordering? |
| `leaderboard_single_hyperparameter_influence_analyze.py` | Rankings CSV | Influence plots | Plots and analyzes the hyperparameter influence data. |
| `workload_reduction.py` | `_TS/*.parquet`, dense workload | Correlation/reduction stats | Analyzes how much the workload can be reduced while maintaining result quality. |
| `similar_strategies.py` | `selected_indices.parquet` | Jaccard correlation heatmaps | Strategy similarity via selected indices — which strategies behave most alike? |
| `strateg_framework_correlation.py` | Strategy metrics | Framework correlation plot | Cross-framework correlation analysis. |

### Leaderboard Variant Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `leaderboard_scenarios.py` | Scenario metrics | Scenario rankings | Ranks strategies under different real-world scenarios (dataset type, start point, hyperparameter variations). |
| `leaderboard_c6_rebuttal.py` | Metric files | Kendall tau correlations, PDFs | Rebuttal analysis with bootstrap confidence intervals for ranking stability. |
| `final_leaderboard_single_cell_correlation.py` | Leaderboard parquets | Correlation stats, plots | Cell-wise correlation analysis within the leaderboard matrix. |
| `analyze_leaderboard_rankings.py` | `plots/leaderboard_invariances/leaderboard_types.csv` | Heatmap correlations | Correlates different leaderboard construction methods. |

### Learning Curve & Example Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `single_learning_curve_example.py` | Sample data | Line plot | Example visualization of a single learning curve. |
| `single_learning_curve_example_auc.py` | Sample data | Line plot with AUC | Example learning curve with AUC annotation. |

### Dataset & Metric Analysis Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `calc_cycle_duration_parquets.py` | Metric CSVs, `05_done_workload.csv` | Threshold plots, duration analysis | Analyzes learning cycle durations and computes duration thresholds. |
| `calculate_dataset_dependend_random_ramp_slope.py` | Selected indices time series | Leaderboard rankings CSV | Computes dataset-dependent random baseline slopes. |
| `dataset_stats.py` | — | — | Dataset statistics (placeholder). |

### Scenario & Real-World Analysis Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `real_world_scenarios_corrs.py` | Scenario metrics CSV | Decomposed correlations | Real-world scenario correlation decomposition. |
| `real_world_scenarios_plots.py` | Scenario data | Scatter/correlation plots | Plots for real-world scenario analysis. |

### Publication & Output Scripts

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `redo_plots_for_paper.py` | All parquet files | Combined ranking plots (PDFs) | Regenerates all publication-ready plots at once. |
| `merge_multiple_plots_single_page.py` | Plot parquets | Merged PDF | Merges multiple parquet-based plots into a single multi-page PDF. |

---

## Important Utility Scripts (`scripts/`)

### Data Preparation Scripts

| Script | Description |
|--------|-------------|
| `scripts/create_dense_workload.py` | Generate a dense workload (all dataset × strategy combinations). |
| `scripts/create_new_extended_dense_workload.py` | Extended version of the dense workload. |
| `scripts/create_gaussian.py` | Generate synthetic Gaussian datasets (balanced/unbalanced). |
| `scripts/create_xor.py` | Download XOR datasets from the LAL project. |
| `scripts/create_auc_selected_ts.py` | Create AUC time series from selected indices data. |
| `scripts/reduce_to_dense.py` | Remove results where the full hyperparameter grid is incomplete, creating a dense grid from sparse experimental results. |

### Conversion Scripts

| Script | Description |
|--------|-------------|
| `scripts/convert_metrics_csvs_to_exp_id_csvs.py` | Reorganize metric CSVs indexed by experiment ID. |
| `scripts/convert_dataset_distances_to_parqet.py` | Convert dataset distance CSV files to parquet format. |
| `scripts/convert_y_pred_to_parquet.py` | Convert y_pred CSV files to parquet format (with timeout handling). |

### Validation Scripts

| Script | Description |
|--------|-------------|
| `scripts/validate_results_schema.py` | Verify that result file formats match the expected schema. |
| `scripts/check_if_exp_ids_are_present.py` | Verify all experiment IDs exist in all metric files. |
| `scripts/find_missing_exp_ids_in_metric_files.py` | Find experiments that are missing from metric CSV files. |
| `scripts/find_broken_file.py` | Identify corrupted or malformed metric CSV files. |
| `scripts/exp_results_data_format_test.py` | Test that result CSV generation and format is correct. |

### Export & Documentation Scripts

| Script | Description |
|--------|-------------|
| `scripts/export_strategy_catalog.py` | Export all AL strategies to JSON/CSV/Markdown with framework info. |
| `scripts/add_github_hyperlinks.py` | Convert file references to GitHub hyperlinks in markdown. |
| `scripts/render_mermaid.py` | Pre-render Mermaid diagrams to SVG for static fallback. |
| `scripts/single_learning_curve.py` | Generate a single example learning curve visualization. |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing `_TS/*.parquet` | These are auto-generated by evaluation scripts. Ensure steps 2–5 completed successfully and `.csv.xz` files exist. |
| Incomplete experiment grid | Use `scripts/reduce_to_dense.py` to remove results where the full hyperparameter grid is incomplete, creating a dense grid from sparse experimental results |
| Missing parquet files | Check `05_done_workload.csv` for completed runs |
| Broken/corrupted metric files | See [Fix Scripts](../personas/reproduce_and_run.md#fix-scripts-only-needed-if-something-breaks) in the Reproduce & Run guide |

---

## Cross-References

- [Architecture & Design](../personas/understand_codebase.md) — Detailed data flow and file dependencies
- [Correlations: Paper ↔ Code](correlations_paper_to_code.md) — Mathematical definitions
- [Analyze OPARA](../personas/analyze_dataset.md) — Research tutorials
- [Reproduce & Run](../personas/reproduce_and_run.md) — Full pipeline including fix scripts
