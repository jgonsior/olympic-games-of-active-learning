# Eva Scripts Reference

Evaluation scripts in [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) for generating analyses and paper figures.

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

## Script Index

| Script | Purpose | Outputs |
|--------|---------|---------|
| `basic_metrics_correlation.py` | Metric correlation matrix | `plots/basic_metrics/*.parquet` |
| `auc_metric_correlation.py` | AUC aggregation correlation | `plots/AUC/*.parquet` |
| `calculate_leaderboard_rankings.py` | Strategy rankings | `plots/final_leaderboard/*.parquet` |
| `final_leaderboard.py` | Leaderboard heatmaps | `plots/final_leaderboard/*.jpg` |
| `learning_curve.py` | Time series generation | `_TS/*.parquet` |
| `single_learning_curve_example.py` | Learning curve figures | `plots/single_learning_curve/*.parquet` |
| `runtime.py` | Runtime analysis | `plots/runtime/*.parquet` |
| `single_hyperparameter_evaluation_metric.py` | Pearson correlations | `plots/single_hyperparameter/*/single_hyper_*.parquet` |
| `single_hyperparameter_evaluation_indices.py` | Jaccard similarities | `plots/single_hyperparameter/*/single_indice_*.parquet` |
| `leaderboard_single_hyperparameter_influence.py` | Hyperparameter rankings | `plots/leaderboard_single_hyperparameter_influence/*.csv` |
| `leaderboard_single_hyperparameter_influence_analyze.py` | Kendall τ analysis | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` |
| `redo_plots_for_paper.py` | Publication PDFs | PDFs in `plots/` |
| `merge_multiple_plots_single_page.py` | Combined figures | Merged PDFs |
| `dataset_stats.py` | Dataset statistics | Console output |

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
