# Strategy Catalog

OGAL benchmarks **76 active learning strategies** across 6 AL frameworks.

---

## Frameworks

| Framework | Enum | Example strategies |
|-----------|------|--------------------|
| ALiPy | `AL_FRAMEWORK.ALIPY` | Random, Uncertainty (LC), CoreSet Greedy |
| libact | `AL_FRAMEWORK.LIBACT` | QUIRE, ALBL |
| small-text | `AL_FRAMEWORK.SMALLTEXT` | Prediction entropy, Breaking ties |
| scikit-activeml | `AL_FRAMEWORK.SKACTIVEML` | Epistemic uncertainty, Expected model change |
| modAL | *(custom runners)* | Margin sampling, Entropy sampling |
| BaaL | *(custom runners)* | BALD |

## Strategy enums

All strategies are defined in `resources/data_types.py` as members of the `AL_STRATEGY` IntEnum.
The mapping from enum to Python class is in `al_strategy_to_python_classes_mapping` in the same file.

```python
from resources.data_types import AL_STRATEGY

AL_STRATEGY.ALIPY_RANDOM          # 1 — Random sampling
AL_STRATEGY.ALIPY_UNCERTAINTY_LC  # 2 — Uncertainty (Least Confident)
AL_STRATEGY.ALIPY_CORESET_GREEDY  # 4 — CoreSet Greedy
# ... 76 total
```

---

**See also:** [Extend OGAL](personas/extend_benchmark.md) (adding a new strategy) · [Paper subset](paper_subset.md)
