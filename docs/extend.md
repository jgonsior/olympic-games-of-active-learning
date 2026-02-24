# Strategies & extending

Frameworks, strategy organization, and the paper experiment grid.

---

## Frameworks

Each AL framework has a dedicated runner in `framework_runners/`:

| Framework | Runner | Example strategies |
|-----------|--------|--------------------|
| ALiPy | `framework_runners/alipy_runner.py` | Random, Uncertainty (LC), CoreSet Greedy |
| libact | `framework_runners/libact_runner.py` | QUIRE, DWUS |
| small-text | `framework_runners/smalltext_runner.py` | Prediction entropy, Breaking ties |
| scikit-activeml | `framework_runners/skactiveml_runner.py` | QBC, Uncertainty sampling |
| Google Playground | `framework_runners/playground_runner.py` | Margin, Mixture, KCenter Greedy |
| OGAL-native | `framework_runners/optimal_runner.py` | OPTIMAL_GREEDY_10, OPTIMAL_GREEDY_20 |

---

## Strategy identification

Every strategy is a member of the `AL_STRATEGY` IntEnum in `resources/data_types.py`.
The mapping from enum to Python class lives in `al_strategy_to_python_classes_mapping` in the same file.
This file is the **source of truth** for all available strategies.

```python
from resources.data_types import AL_STRATEGY

AL_STRATEGY.ALIPY_RANDOM          # 1
AL_STRATEGY.ALIPY_UNCERTAINTY_LC  # 2
AL_STRATEGY.ALIPY_CORESET_GREEDY  # 4
# ... see AL_STRATEGY for the full list
```

---

## Adding a strategy

1. Add a new member to `AL_STRATEGY` in `resources/data_types.py` with an unused integer ID.
2. Map it to a Python class in `al_strategy_to_python_classes_mapping`.
3. Reference it in your `resources/exp_config.yaml` block under `EXP_GRID_STRATEGY`.

---

## Paper subset

The paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)) uses the **`full_exp_jan`** config block in `resources/exp_config.yaml`.

### Experiment grid

| Dimension | Values | Count |
|-----------|--------|-------|
| Strategies | see list below | 28 |
| Datasets | 92 tabular classification datasets (OpenML + Kaggle) | 92 |
| Learner models | RF, RBF_SVM, MLP | 3 |
| Batch sizes | 1, 5, 10, 20, 50, 100 | 6 |
| Train/test splits | 0–4 | 5 |
| Start sets | 0–19 | 20 |
| Random seeds | 0 | 1 |

**Total: 4,636,800 experiments** (92 × 28 × 3 × 6 × 5 × 20 × 1).
Runtime limit: **5 minutes (300 s) per AL cycle**.

Released results: [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

### Strategies used in the paper

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
| small-text | `SMALLTEXT_CONTRASTIVEAL` |
| small-text | `SMALLTEXT_RANDOM` |
| scikit-activeml | `SKACTIVEML_QBC` |
| scikit-activeml | `SKACTIVEML_US_MARGIN` |
| scikit-activeml | `SKACTIVEML_US_LC` |
| scikit-activeml | `SKACTIVEML_US_ENTROPY` |
| scikit-activeml | `SKACTIVEML_COST_EMBEDDING` |
| scikit-activeml | `SKACTIVEML_QBC_VOTE_ENTROPY` |
| scikit-activeml | `SKACTIVEML_QUIRE` |

All enums are defined in `resources/data_types.py`.
