# Paper Subset

What was used in the paper [arXiv:2506.03817](https://arxiv.org/abs/2506.03817).

---

## Config

The paper run is defined as **`full_exp_jan`** in `resources/exp_config.yaml`.

## Experiment grid

| Dimension | Values | Count |
|-----------|--------|-------|
| Strategies | see list below | 31 |
| Datasets | 93 tabular classification datasets (OpenML + Kaggle) | 93 |
| Learner models | RF, RBF_SVM, MLP | 3 |
| Batch sizes | 1, 5, 10, 20, 50, 100 | 6 |
| AL cycles | 100 | 1 |
| Train/test splits | 0–4 | 5 |
| Start sets | 0–19 | 20 |
| Random seeds | 0 | 1 |

Total: ~4.6M experiments (~3.6M CPU hours on HPC).

## Strategies used in the paper

| Framework | Strategy enum |
|-----------|---------------|
| ALiPy | `ALIPY_RANDOM` |
| ALiPy | `ALIPY_UNCERTAINTY_LC` |
| ALiPy | `ALIPY_UNCERTAINTY_MM` |
| ALiPy | `ALIPY_UNCERTAINTY_ENTROPY` |
| ALiPy | `ALIPY_GRAPH_DENSITY` |
| ALiPy | `ALIPY_CORESET_GREEDY` |
| ALiPy | `ALIPY_DENSITY_WEIGHTED` |
| OGAL-native | `OPTIMAL_GREEDY_10` |
| OGAL-native | `OPTIMAL_GREEDY_20` |
| libact | `LIBACT_UNCERTAINTY_LC` |
| libact | `LIBACT_UNCERTAINTY_SM` |
| libact | `LIBACT_UNCERTAINTY_ENT` |
| libact | `LIBACT_DWUS` |
| libact | `LIBACT_QUIRE` |
| small-text | `SMALLTEXT_LEASTCONFIDENCE` |
| small-text | `SMALLTEXT_PREDICTIONENTROPY` |
| small-text | `SMALLTEXT_BREAKINGTIES` |
| small-text | `SMALLTEXT_EMBEDDINGKMEANS` |
| small-text | `SMALLTEXT_GREEDYCORESET` |
| small-text | `SMALLTEXT_LIGHTWEIGHTCORESET` |
| small-text | `SMALLTEXT_CONTRASTIVEAL` |
| small-text | `SMALLTEXT_RANDOM` |
| scikit-activeml | `SKACTIVEML_QBC` |
| scikit-activeml | `SKACTIVEML_US_MARGIN` |
| scikit-activeml | `SKACTIVEML_US_LC` |
| scikit-activeml | `SKACTIVEML_US_ENTROPY` |
| scikit-activeml | `SKACTIVEML_EXPECTED_AVERAGE_PRECISION` |
| scikit-activeml | `SKACTIVEML_COST_EMBEDDING` |
| scikit-activeml | `SKACTIVEML_DAL` |
| scikit-activeml | `SKACTIVEML_QBC_VOTE_ENTROPY` |
| scikit-activeml | `SKACTIVEML_QUIRE` |

All strategy enums are defined in `resources/data_types.py` (`AL_STRATEGY` IntEnum).

## Key figures

| Figure / Table | Evaluation script |
|---------------|-------------------|
| Table 1 — Strategy leaderboard | `eva_scripts.final_leaderboard` |
| Pearson-$r$ heatmaps | `eva_scripts.workload_reduction` |
| Jaccard-$J$ heatmaps | `eva_scripts.single_hyperparameter_evaluation_indices` |
| Kendall-$\tau_b$ heatmaps | `eva_scripts.leaderboard_single_hyperparameter_influence` |

---

**See also:** [Evaluation scripts](evaluation_scripts.md) · [Strategy catalog](strategy_catalog.md) · [Results format](results_format.md)
