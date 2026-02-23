# Paper Subset

What was used in the paper [arXiv:2506.03817](https://arxiv.org/abs/2506.03817).

---

## Experiment grid

The paper's main experiment grid is defined as `full_exp_jan` in `resources/exp_config.yaml`:

| Dimension | Values | Count |
|-----------|--------|-------|
| Strategies | 28 (subset of 76) | 28 |
| Datasets | 92 | 92 |
| Learner models | RF, RBF_SVM, MLP | 3 |
| Batch sizes | 1, 5, 10, 20, 50, 100 | 6 |
| Train/test splits | 0–4 | 5 |
| Start sets | 0–19 | 20 |
| Random seeds | 1 | 1 |

Total: ~4.6M experiments.

## Key figures

| Figure / Table | Evaluation script |
|---------------|-------------------|
| Table 1 — Strategy leaderboard | `eva_scripts.final_leaderboard` |
| Pearson-$r$ heatmaps | `eva_scripts.workload_reduction` |
| Jaccard-$J$ heatmaps | `eva_scripts.single_hyperparameter_evaluation_indices` |
| Kendall-$\tau_b$ heatmaps | `eva_scripts.leaderboard_single_hyperparameter_influence` |

---

**See also:** [Evaluation scripts](evaluation_scripts.md) · [Strategy catalog](strategy_catalog.md) · [Results format](results_format.md)
