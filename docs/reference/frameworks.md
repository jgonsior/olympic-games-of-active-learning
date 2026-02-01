# Frameworks Reference

OGAL integrates 5 AL frameworks under a **unified wrapper** that runs strategies consistently.

---

## Framework Summary

| Framework | Strategies | Upstream | OGAL Fork | Adapter |
|-----------|------------|----------|-----------|---------|
| ALiPy | 14 | [NUAA-AL/ALiPy](https://github.com/NUAA-AL/ALiPy) | [jgonsior/ALiPy](https://github.com/jgonsior/ALiPy) | `alipy_runner.py` |
| libact | 9 | [ntucllab/libact](https://github.com/ntucllab/libact) | [jgonsior/libact](https://github.com/jgonsior/libact) | `libact_runner.py` |
| small-text | 9 | [webis-de/small-text](https://github.com/webis-de/small-text) | None (PyPI) | `smalltext_runner.py` |
| scikit-activeml | 14 | [scikit-activeml/scikit-activeml](https://github.com/scikit-activeml/scikit-activeml) | [jgonsior/scikit-activeml](https://github.com/jgonsior/scikit-activeml) | `skactiveml_runner.py` |
| Playground | 9 | [google/active-learning](https://github.com/google/active-learning) | [jgonsior/active-learning](https://github.com/jgonsior/active-learning) | `playground_runner.py` |
| OPTIMAL | 4 | N/A (OGAL-native) | N/A | `optimal_runner.py` |

---

## What's Unified

- **Consistent protocol**: Same train/test splits, seeds, budgets, and metrics across all frameworks
- **Unified output schema**: All strategies produce the same output format
- **Framework-agnostic configuration**: Configure by strategy name, not framework details

```mermaid
flowchart LR
    subgraph Config["Configuration"]
        C1["exp_config.yaml"]
    end
    
    subgraph Unified["OGAL Unified Runner"]
        U1["02_run_experiment.py"]
        U2["base_runner.py"]
    end
    
    subgraph Adapters["Framework Adapters"]
        A1["alipy_runner.py"]
        A2["libact_runner.py"]
        A3["smalltext_runner.py"]
        A4["skactiveml_runner.py"]
        A5["playground_runner.py"]
        A6["optimal_runner.py"]
    end
    
    subgraph Frameworks["Backend Frameworks"]
        F1["ALiPy"]
        F2["libact"]
        F3["small-text"]
        F4["scikit-activeml"]
        F5["Playground"]
        F6["OGAL Optimal"]
    end
    
    C1 --> U1
    U1 --> U2
    U2 --> A1 & A2 & A3 & A4 & A5 & A6
    A1 --> F1
    A2 --> F2
    A3 --> F3
    A4 --> F4
    A5 --> F5
    A6 --> F6
```

---

## Why Forks?

OGAL uses forks of ALiPy, libact, scikit-activeml, and Playground to ensure:

- Python 3.11 compatibility
- Dependency conflict resolution
- API compatibility with the experiment protocol

small-text is used directly from PyPI without modifications.

---

## Cross-References

- **[Strategy Catalog](strategy_catalog.md)**: Complete list of all strategies (strategy-first)
- **[Runbook](runbook.md)**: How experiments are executed
- **[Architecture in 10 Minutes](concepts/architecture_in_10_minutes.md)**: Configuration options
