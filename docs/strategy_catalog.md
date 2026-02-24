# Strategy Catalog

OGAL benchmarks many active learning strategies across multiple AL frameworks.

---

## Glossary

- **Framework** — an external AL library (e.g., ALiPy, scikit-activeml) or an OGAL-native baseline. Each framework has a runner in `framework_runners/`.
- **Strategy enum / ID** — each strategy is a member of the `AL_STRATEGY` IntEnum in `resources/data_types.py`. The integer ID uniquely identifies the strategy across all frameworks.
- **Paper subset** — the 28 strategies actually used in the paper run. See [Paper Run Definition](paper_subset.md) for the exact list.

## Frameworks

| Framework | Runner | Example strategies |
|-----------|--------|--------------------|
| ALiPy | `framework_runners/alipy_runner.py` | Random, Uncertainty (LC), CoreSet Greedy |
| libact | `framework_runners/libact_runner.py` | QUIRE, DWUS |
| small-text | `framework_runners/smalltext_runner.py` | Prediction entropy, Breaking ties |
| scikit-activeml | `framework_runners/skactiveml_runner.py` | QBC, Uncertainty sampling |
| Google Playground | `framework_runners/playground_runner.py` | Margin, Mixture, KCenter Greedy |
| OGAL-native | `framework_runners/optimal_runner.py` | OPTIMAL_GREEDY_10, OPTIMAL_GREEDY_20 |

## Strategy enums

All strategies are defined in `resources/data_types.py` as members of the `AL_STRATEGY` IntEnum.
The mapping from enum to Python class is in `al_strategy_to_python_classes_mapping` in the same file.
This file is the **source of truth** for the full list of available strategies.

```python
from resources.data_types import AL_STRATEGY

AL_STRATEGY.ALIPY_RANDOM          # 1 — Random sampling
AL_STRATEGY.ALIPY_UNCERTAINTY_LC  # 2 — Uncertainty (Least Confident)
AL_STRATEGY.ALIPY_CORESET_GREEDY  # 4 — CoreSet Greedy
# ... see AL_STRATEGY enum for the full list
```

## Paper subset vs full catalog

The repository contains more strategies than were used in the paper.
For the exact list of 28 strategies (and the full experiment grid) used in the
paper run, see **[Paper Run Definition](paper_subset.md)**.

---

**See also:** [Extend OGAL](personas/extend_benchmark.md) (adding a new strategy) · [Paper Run Definition](paper_subset.md)
