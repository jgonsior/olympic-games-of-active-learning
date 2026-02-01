# Eva Scripts Reference

Evaluation scripts in [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) for generating analyses and paper figures.

---

## 5 Canonical Recipes

The scripts most researchers need:

### 1. Generate Leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

### 2. Generate Learning Curves

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

**Output:** `_TS/*.parquet`, `plots/single_learning_curve/*.parquet`

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

## Compact Script Index

| Script | Produces | Reads | Typical Use |
|--------|----------|-------|-------------|
| `learning_curve.py` | `_TS/*.parquet` | Per-cycle CSVs | **Required first** for most scripts |
| `final_leaderboard.py` | Rank heatmaps, leaderboard | `_TS/*.parquet` | Main results |
| `single_hyperparameter_evaluation_metric.py` | Blue heatmaps | `_TS/*.parquet` | Hyperparameter sensitivity |
| `single_hyperparameter_evaluation_indices.py` | Green heatmaps | `selected_indices.parquet` | Sample selection analysis |
| `leaderboard_single_hyperparameter_influence.py` | Kendall τ data | `_TS/*.parquet` | Ranking stability |
| `runtime.py` | Runtime charts | `query_selection_time.csv` | Performance analysis |
| `basic_metrics_correlation.py` | Metric correlation matrix | Per-cycle CSVs | Metric comparison |
| `redo_plots_for_paper.py` | PDFs | All parquet files | Publication figures |

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
- [Analyze the Dataset](../analyze_opara.md) — Research tutorials
