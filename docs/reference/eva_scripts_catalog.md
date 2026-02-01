# Eva Scripts (top 6 + index)

Six most-used scripts and what they produce:

| Script | Purpose | Outputs |
|--------|---------|---------|
| `basic_metrics_correlation.py` | Correlate standard metrics | `plots/basic_metrics/Standard Metrics.parquet` |
| `auc_metric_correlation.py` | Correlate AUC aggregations | `plots/AUC/auc_*.parquet` |
| `calculate_leaderboard_rankings.py` | Strategy rankings | `plots/final_leaderboard/*.parquet`, `plots/leaderboard_invariances/*.csv` |
| `final_leaderboard.py` | Publication-ready leaderboards | `plots/final_leaderboard/rank_*.parquet`, `*.jpg` |
| `runtime.py` | Strategy runtime | `plots/runtime/query_selection_time.parquet` |
| `learning_curve.py` | Aggregated learning curves | `_TS/*.parquet`, `plots/single_learning_curve/*.parquet` |

---

## Quick recipes

### Leaderboard

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

### Learning curves

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

### Three correlations (paper)

```bash
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

---

## Index (other scripts)

| Script | Purpose | Outputs |
|--------|---------|---------|
| `single_learning_curve_example.py` | Single dataset curve | `plots/single_learning_curve/weighted_f1-score.parquet` |
| `redo_plots_for_paper.py` | Re-render PDFs | PDFs in `plots/` |
| `merge_multiple_plots_single_page.py` | Merge heatmaps | Merged PDFs |
| `dataset_stats.py` | Dataset stats | Console |
| `single_hyperparameter_evaluation_metric.py` | Metric corr by hyperparam | `plots/single_hyperparameter/*/single_hyper_*.parquet` |
| `single_hyperparameter_evaluation_indices.py` | Jaccard of queried samples | `plots/single_hyperparameter/*/single_indice_*.parquet` |
| `leaderboard_single_hyperparameter_influence.py` | Ranking invariance | `plots/leaderboard_single_hyperparameter_influence/*.csv` |
| `leaderboard_single_hyperparameter_influence_analyze.py` | Kendall τ | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` |

---

## Pointers

- Inputs missing? Run `python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan`.
- Correlation math: see [Correlations: Paper ↔ Code](correlations_paper_to_code.md).
- Schemas: [Results schema](results_schema.md).
