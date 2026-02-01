# Analyze OPARA

**You want to mine the 4.6M pre-computed experiment results for your own research—no experiments needed.**

---

## 10-Minute First Win

### 1. Get the Data

```bash
wget <URL_FROM_DOI_LANDING_PAGE>   # From DOI:10.25532/OPARA-862
unzip full_exp_jan.zip -d /path/to/results/
```

### 2. Setup Environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
```

### 3. Generate Leaderboard

```bash
export OGAL_OUTPUT=/path/to/results
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

### 4. Load and Explore

```python
import pandas as pd

# Load leaderboard
lb = pd.read_parquet("/path/to/results/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet")
print("Top strategies:", lb.mean(axis=0).sort_values().head(5))

# Load completed experiments
done = pd.read_csv("/path/to/results/full_exp_jan/05_done_workload.csv")
print(f"Total experiments: {len(done):,}")
```

---

## Starter Recipe A: Compare Strategies Across Datasets

```python
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load time series (run learning_curve first if missing)
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")

# Average performance by strategy
by_strategy = ts.groupby('EXP_STRATEGY')['metric_value'].agg(['mean', 'std'])
print(by_strategy.sort_values('mean', ascending=False).head(10))

# Performance by batch size
by_batch = ts.groupby('EXP_BATCH_SIZE')['metric_value'].mean()
print(by_batch)
```

**If `_TS/*.parquet` is missing:**

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

---

## Starter Recipe B: Load Per-Cycle Metrics

```python
# Load per-cycle accuracy for a specific strategy/dataset
accuracy = pd.read_csv(
    f"{OGAL_OUTPUT}/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv.xz",
    compression='xz'
)

# Join with workload to get hyperparameters
done = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv")
merged = done.merge(accuracy, on="EXP_UNIQUE_ID")
print(merged.head())
```

---

## Key Files

| File | What It Contains |
|------|------------------|
| `05_done_workload.csv` | 4.6M completed experiments (workload index) |
| `<STRATEGY>/<DATASET>/accuracy.csv.xz` | Per-cycle accuracy |
| `<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` | Per-cycle F1 |
| `_TS/*.parquet` | Aggregated time series (generated) |
| `plots/final_leaderboard/*.parquet` | Strategy rankings (generated) |

??? info "Key Columns"
    | Column | Description |
    |--------|-------------|
    | `EXP_UNIQUE_ID` | Primary key |
    | `EXP_DATASET` | Dataset enum (3=Iris, etc.) |
    | `EXP_STRATEGY` | Strategy enum (7=ALIPY_RANDOM, etc.) |
    | `EXP_LEARNER_MODEL` | 1=RF, 2=MLP, 3=SVM |
    | `EXP_BATCH_SIZE` | 1, 5, 10, 20, 50, 100 |

---

## Next Steps

| Goal | Page |
|------|------|
| Reproduce paper figures | [Reproduce the Paper](reproduce_paper.md) |
| Research ideas from the data | [Research Ideas](research_ideas.md) |
| Understand correlation metrics | [Correlations: Paper ↔ Code](../reference/correlations_paper_to_code.md) |
