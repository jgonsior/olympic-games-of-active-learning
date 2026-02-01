# Analyze the Dataset

**You're a researcher who wants to use the 4.6M pre-computed experiment results for your own research—without running experiments yourself.**

This guide shows you how to load, parse, and analyze the OPARA archived dataset.

---

## What You'll Accomplish

- Download and extract the archived experiment results
- Understand the data schema and file structure
- Load data into pandas for analysis
- Run basic queries and generate insights

---

## Prerequisites

- Basic Python/pandas knowledge
- ~50-100 GB disk space (for full archive, or less for subsets)

---

## Step 1: Get the Data

### 1.1 Download from OPARA

**Canonical source:** [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

```bash
# Download from the DOI landing page
wget <URL_FROM_DOI_LANDING_PAGE>

# Extract
export OGAL_OUTPUT=/path/to/results
unzip full_exp_jan.zip -d ${OGAL_OUTPUT}/
```

### 1.2 Verify the Archive

```bash
# Check the structure
ls ${OGAL_OUTPUT}/full_exp_jan/
# Should show: 05_done_workload.csv, strategy directories, etc.

# Check experiment count
wc -l ${OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv
# Should show ~4.6M lines
```

---

## Step 2: Setup the Environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning

# Install dependencies
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# Configure paths
cat > .server_access_credentials.cfg << EOF
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF
```

---

## Step 3: Understand the Data

### Archive Layout

```
full_exp_jan/
├── 05_done_workload.csv          # 4.6M completed experiments (workload index)
├── 01_workload.csv               # Full hyperparameter grid
├── <STRATEGY>/<DATASET>/
│   ├── accuracy.csv.xz           # Per-cycle accuracy
│   ├── weighted_f1-score.csv.xz  # Per-cycle F1
│   ├── selected_indices.csv.xz   # Queried sample indices
│   └── full_auc_*.csv.xz         # Aggregated AUC metrics
└── _TS/                          # Time series parquets (generated)
```

### Key Columns

| Column | Description | Example Values |
|--------|-------------|----------------|
| `EXP_UNIQUE_ID` | Primary key for each experiment | 0, 1, 2, ... |
| `EXP_DATASET` | Dataset enum value | 3 (Iris), 45 (wine_origin) |
| `EXP_STRATEGY` | AL strategy enum value | 7 (ALIPY_RANDOM) |
| `EXP_LEARNER_MODEL` | Learner model | 1 (RF), 2 (MLP), 3 (SVM) |
| `EXP_BATCH_SIZE` | Query batch size | 1, 5, 10, 20, 50, 100 |

### Paper Terminology

The paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)) defines experiments as **E = (𝒮, D, 𝒯, ℐ, M, b, c, ℒ)**:

| Symbol | Term | Values in Archive |
|--------|------|-------------------|
| 𝕊 | AL Strategy | 28 strategies |
| 𝔻 | Dataset | 92 datasets |
| 𝕃 | Learner | RF, MLP, SVM |
| 𝔹 | Batch Size | 1, 5, 10, 20, 50, 100 |
| 𝕋 | Train-Test Split | 5 per dataset |
| 𝕀 | Start Set | 20 per split |

---

## Step 4: Load and Query the Data

### 4.1 Load the Workload Index

```python
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load completed experiments
done = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv")
print(f"Total experiments: {len(done):,}")
print(done.columns.tolist())
```

### 4.2 Load Metric Results

```python
# Load per-cycle accuracy for a specific strategy/dataset
accuracy = pd.read_csv(
    f"{OGAL_OUTPUT}/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv.xz",
    compression='xz'
)

# Join with workload to get hyperparameters
merged = done.merge(accuracy, on="EXP_UNIQUE_ID")
print(merged.head())
```

### 4.3 Load Time Series Data

```python
# After running eva_scripts.learning_curve (see below)
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")
print(ts.groupby(['EXP_STRATEGY'])['metric_value'].mean().sort_values(ascending=False).head(10))
```

---

## Step 5: Generate Key Artifacts

### 5.1 Generate Time Series (Required First)

Most analysis scripts require time series parquet files:

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

**Output:** `_TS/*.parquet` files

### 5.2 Generate Leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

### 5.3 Load and Analyze the Leaderboard

```python
leaderboard = pd.read_parquet(
    f"{OGAL_OUTPUT}/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet"
)

# Top strategies by average rank
print(leaderboard.mean(axis=0).sort_values().head(10))
```

---

## Common Analysis Patterns

### Group by Strategy

```python
# Average performance by strategy
by_strategy = ts.groupby('EXP_STRATEGY')['metric_value'].agg(['mean', 'std', 'count'])
print(by_strategy.sort_values('mean', ascending=False))
```

### Filter by Dataset

```python
# Filter to specific datasets
iris_only = ts[ts['EXP_DATASET'] == 3]  # Iris = 3
print(iris_only.groupby('EXP_STRATEGY')['metric_value'].mean())
```

### Compare Batch Sizes

```python
# Performance by batch size
by_batch = ts.groupby('EXP_BATCH_SIZE')['metric_value'].mean()
print(by_batch)
```

---

## Next Steps

| Goal | Page |
|------|------|
| Reproduce paper figures exactly | [Reproduce Paper Results](reproduce_paper.md) |
| Understand the correlation metrics | [Correlations: Paper ↔ Code](../reference/correlations_paper_to_code.md) |
| Get research ideas from the data | [Research Ideas](research_ideas.md) |
| Understand output file schemas | [Results Schema](../reference/results_schema.md) |
| Learn about all evaluation scripts | [Eva Scripts Catalog](../reference/eva_scripts_catalog.md) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing `_TS/*.parquet` files | Run `python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan` first |
| Memory errors | Use `scripts/reduce_to_dense.py` for a smaller subset |
| Cannot find strategy names | See [Strategy Catalog](../reference/strategy_catalog.md) for enum mappings |
| Cannot find dataset names | Check `resources/data_types.py` for DATASET enum |
