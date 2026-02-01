# Strategy Catalog

OGAL provides 50+ AL strategies from 5 frameworks under a unified API.

## Strategy Dispatch Flow

How OGAL dispatches a workload item to the correct strategy:

```mermaid
flowchart TD
    subgraph Input["Workload Configuration"]
        W1[EXP_STRATEGY enum value]
        W2[EXP_LEARNER_MODEL]
        W3[EXP_DATASET]
        W4[EXP_BATCH_SIZE]
    end
    
    subgraph Registry["Strategy Registry"]
        R1["resources/data_types.py<br/>AL_STRATEGY enum"]
        R2["al_strategy_to_python_classes_mapping"]
        R3["AL_framework_to_classes_mapping"]
    end
    
    subgraph Dispatch["Unified Runner"]
        D1["02_run_experiment.py"]
        D2["Detect framework from strategy prefix"]
        D3["Instantiate framework adapter"]
    end
    
    subgraph Adapters["Framework Adapters"]
        A1["ALIPY_AL_Experiment"]
        A2["LIBACT_Experiment"]
        A3["SMALLTEXT_AL_Experiment"]
        A4["SKACTIVEML_AL_Experiment"]
        A5["PLAYGROUND_AL_Experiment"]
        A6["OPTIMAL_AL_Experiment"]
    end
    
    subgraph Execution["AL Loop"]
        E1["Initialize learner model"]
        E2["Query strategy for samples"]
        E3["Update labeled set"]
        E4["Retrain model"]
        E5["Record metrics"]
    end
    
    subgraph Output["Results"]
        O1["Per-cycle metrics CSV"]
        O2["Selected indices"]
        O3["Timing data"]
    end
    
    W1 --> R1
    R1 --> R2
    R2 --> D1
    D1 --> D2
    D2 --> R3
    R3 --> D3
    D3 --> A1 & A2 & A3 & A4 & A5 & A6
    A1 & A2 & A3 & A4 & A5 & A6 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> E5
    E5 --> O1 & O2 & O3
```

---

## Quick Reference

| Family | Examples | Frameworks |
|--------|----------|------------|
| **Uncertainty** | Least Confident, Entropy, Margin | ALiPy, libact, small-text, scikit-activeml |
| **Committee** | Query by Committee | ALiPy, libact, scikit-activeml |
| **Diversity/Density** | Coreset, K-Center, Density-weighted | ALiPy, libact, small-text, Playground |
| **Hybrid/Meta** | ALBL, Contrastive AL, Discriminative AL | libact, small-text, scikit-activeml |
| **Expected Error** | EER, Monte Carlo EER | ALiPy, libact, scikit-activeml |
| **Oracle** | Greedy Optimal, True Optimal | OGAL native |

## Using Strategies

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_STRATEGY:
    - ALIPY_UNCERTAINTY_LC    # Least Confident (ALiPy)
    - LIBACT_QUIRE            # QUIRE (libact)
    - SMALLTEXT_EMBEDDINGKMEANS  # Embedding K-Means (small-text)
    - SKACTIVEML_QBC          # Query by Committee (scikit-activeml)
```

## Export Full Catalog

```bash
# Generate complete strategy list
python -c "
from resources.data_types import AL_STRATEGY, al_strategy_to_python_classes_mapping
import csv
with open('strategy_catalog.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'Strategy', 'Framework'])
    for s in AL_STRATEGY:
        fw = s.name.split('_')[0]
        writer.writerow([s.value, s.name, fw])
print('Exported to strategy_catalog.csv')
"
```

## Strategy Constraints

| Constraint | Strategies |
|------------|------------|
| **Binary-only** | `ALIPY_LAL`, `ALIPY_UNCERTAINTY_DTB`, `ALIPY_BMDR`, `ALIPY_SPAL` |
| **Not HPC-suitable** | `ALIPY_LAL` |

## Cross-References

- [Frameworks](frameworks.md) — Backend framework details
- [Results Schema](results_schema.md) — Output file formats
- [Runbook](runbook.md) — Running experiments
