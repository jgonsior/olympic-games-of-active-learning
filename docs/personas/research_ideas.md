# Research Ideas

**Open questions and unexplored directions using OGAL's 4.6M experiments.**

---

## 10 Research Directions

| Idea | What OGAL Enables | Key Files |
|------|-------------------|-----------|
| **1. Stopping point analysis** | Per-cycle metrics reveal when AL should stop | `_TS/weighted_f1-score.parquet` |
| **2. Strategy recommendation (meta-learning)** | Predict best strategy from dataset features | `_/<CATEGORIZER>/<DATASET>.npz` |
| **3. Sample selection patterns** | Which samples do strategies pick? When does it matter? | `selected_indices.csv.xz` |
| **4. Dataset difficulty taxonomy** | Cluster datasets by AL behavior | `plots/final_leaderboard/*.parquet` |
| **5. Hyperparameter sensitivity** | Which hyperparameters matter? Interactions? | `_TS/*.parquet` + groupby |
| **6. Failure mode discovery** | When/why do strategies fail? | `05_failed_workloads.csv` |
| **7. Budget allocation** | Variable batch sizes during AL | Per-cycle metrics |
| **8. Early stopping prediction** | Can first 5 iterations predict final performance? | Per-cycle CSVs |
| **9. Learner-strategy interactions** | Do strategies behave differently with RF vs MLP vs SVM? | Time series grouped by model |
| **10. Cross-dataset generalization** | Which strategies transfer best? | 92 datasets × 28 strategies |

---

## Quick Start Template

```python
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load experiments
done = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv")

# Load time series
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")

# Load leaderboard
lb = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet")

# Your analysis here...
```

---

## Citation

```bibtex
@article{gonsior2025ogal,
  title={{Olympic Games of Active Learning}},
  author={Gonsior, Julius and others},
  journal={arXiv preprint arXiv:2506.03817},
  year={2025}
}
```

Dataset DOI: [10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

---

## Next Steps

| Goal | Page |
|------|------|
| Load the data | [Analyze OPARA](analyze_dataset.md) |
| Run your own experiments | [Extend the Benchmark](extend_benchmark.md) |
