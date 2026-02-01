# Eva Scripts Reference

Evaluation scripts in [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) for generating analyses and paper figures.

---

## Script Catalog Overview

| Script | Purpose | Required Inputs | Outputs | Paper Reference |
|--------|---------|-----------------|---------|-----------------|
| `basic_metrics_correlation.py` | Correlation matrix of standard ML metrics | Per-cycle metric CSVs, `05_done_workload.csv` | `plots/basic_metrics/Standard Metrics.parquet` | Metric comparison |
| `auc_metric_correlation.py` | Correlation of AUC-based aggregation metrics | AUC metric files (`full_auc_*.csv.xz`) | `plots/AUC/auc_*.parquet` | Aggregation comparison |
| `calculate_leaderboard_rankings.py` | Strategy rankings across datasets | Time series parquets in `_TS/` | `plots/final_leaderboard/*.parquet`, `plots/leaderboard_invariances/*.csv` | Leaderboard tables |
| `final_leaderboard.py` | Publication-quality final leaderboard heatmaps | `_TS/*.parquet` | `plots/final_leaderboard/rank_*.parquet`, `*.jpg` | Main results |
| `leaderboard_single_hyperparameter_influence.py` | Impact of single hyperparameter on leaderboard | `_TS/*.parquet` | `plots/leaderboard_single_hyperparameter_influence/*.csv` | RQ1 analysis |
| `leaderboard_single_hyperparameter_influence_analyze.py` | Kendall τ analysis of hyperparameter influence | Above CSV files | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` | RQ1 heatmaps |
| `single_hyperparameter_evaluation_metric.py` | Correlation of metrics across hyperparameter values | `_TS/*.parquet` | `plots/single_hyperparameter/*/single_hyper_*.parquet` | Blue heatmaps |
| `single_hyperparameter_evaluation_indices.py` | Jaccard similarity of queried samples | `_TS/selected_indices.parquet` | `plots/single_hyperparameter/*/single_indice_*.parquet` | Green heatmaps |
| `runtime.py` | Strategy runtime analysis | `query_selection_time.csv` files | `plots/runtime/query_selection_time.parquet` | Runtime bar chart |
| `learning_curve.py` | Aggregated learning curves | Per-cycle metric CSVs | `plots/single_learning_curve/*.parquet` | Learning curve figures |
| `single_learning_curve_example.py` | Single dataset learning curve example | `_TS/weighted_f1-score.parquet` | `plots/single_learning_curve/weighted_f1-score.parquet`, `single_exemplary_learning_curve.parquet` | Fig. 2-3 |
| `redo_plots_for_paper.py` | Re-render publication-quality PDFs | All above parquet files | PDFs in `plots/` subdirectories | All paper figures |
| `merge_multiple_plots_single_page.py` | Combine heatmaps into single figures | Individual PDFs | Merged PDFs | Combined figures |
| `dataset_stats.py` | Dataset statistics | None | Console output | Dataset table |

---

## Canonical Recipes

### 1. Generate Leaderboard

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

### 2. Generate Learning Curves

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

**Output:** `_TS/*.parquet`, `plots/single_learning_curve/*.parquet`

### 3. Compute Three Correlations (Paper)

```bash
# Metric-based (Pearson) → Blue heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan

# Queried samples (Jaccard) → Green heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan

# Ranking invariance (Kendall) → Orange heatmaps
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

### 4. Publication-Ready Plots

```bash
python -m eva_scripts.redo_plots_for_paper --EXP_TITLE full_exp_jan
```

---

## Prerequisites

Most scripts require time series data in `_TS/`. If missing:

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing `_TS/*.parquet` | Run `learning_curve.py` first |
| Memory errors | Use `scripts/reduce_to_dense.py` for smaller subset |
| Missing parquet files | Check `05_done_workload.csv` for completed runs |

---

## Cross-References

- [Correlations: Paper ↔ Code](correlations_paper_to_code.md) — Mathematical definitions
- [Results Schema](results_schema.md) — Output file formats
- [Analyze the Dataset](../analyze_dataset.md) — Research tutorials
