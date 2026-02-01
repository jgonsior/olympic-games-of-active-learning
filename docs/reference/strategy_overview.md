# Strategy overview

Snapshot of OGAL strategy families and where to find the full catalog.

- **Families:** Uncertainty, Committee, Diversity/Density, Hybrid/Meta, Expected Error, Oracle
- **Frameworks:** ALiPy, libact, small-text, scikit-activeml, Playground, OGAL Optimal

## Top-level view

| Family | Examples | Frameworks |
|--------|----------|------------|
| Uncertainty | Least Confident, Entropy, Margin | ALiPy, libact, small-text, scikit-activeml |
| Committee | Query by Committee | ALiPy, libact, scikit-activeml |
| Diversity/Density | Coreset, K-Center, Density-weighted | ALiPy, libact, small-text, Playground |
| Hybrid/Meta | ALBL, Contrastive AL, Discriminative AL | libact, small-text, scikit-activeml |
| Expected Error | EER, Monte Carlo EER | ALiPy, libact, scikit-activeml |
| Oracle | Greedy Optimal, True Optimal | OGAL native |

## Full list (generated)

Export the complete list (ID, name, framework):

```bash
python - << 'EOF'
import csv
from resources.data_types import AL_STRATEGY
with open('strategy_catalog.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['ID','Strategy','Framework'])
    for s in AL_STRATEGY:
        w.writerow([s.value, s.name, s.name.split('_')[0]])
print("Wrote strategy_catalog.csv")
EOF
```

Upload `strategy_catalog.csv` alongside your docs build or serve it from CI as a downloadable artifact.

## Cross-references

- Framework details: [Frameworks](frameworks.md)
- Results schema: [Results schema](results_schema.md)
- Run experiments: [Runbook](runbook.md)
