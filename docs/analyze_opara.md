# Analyze OPARA

Use the published OPARA dataset (4.6M experiments) for your own research — without rerunning anything.

---

## Setup

### 1. Get the Archived Data

**Canonical source:** [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

```bash
wget <URL_FROM_DOI_LANDING_PAGE>
unzip full_exp_jan.zip -d /path/to/results/full_exp_jan
```

### 2. Setup OGAL Environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
```

### 3. Configure Paths

```bash
export OGAL_OUTPUT=/path/to/results
cat > .server_access_credentials.cfg << EOF
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF
```

---

## Background

### Paper Terminology

The paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)) defines experiments as **E = (𝒮, D, 𝒯, ℐ, M, b, c, ℒ)**:

| Symbol | Term | Values |
|--------|------|--------|
| 𝕊 | AL Strategy | 28 strategies |
| 𝔻 | Dataset | 92 datasets |
| 𝕃 | Learner | RF, MLP, SVM |
| 𝔹 | Batch Size | 1, 5, 10, 20, 50, 100 |
| 𝕋 | Train-Test Split | 5 per dataset |
| 𝕀 | Start Set | 20 per split |

### Archive Layout

```
full_exp_jan/
├── <STRATEGY>/<DATASET>/
│   ├── accuracy.csv.xz           # Per-cycle accuracy
│   ├── weighted_f1-score.csv.xz  # Per-cycle F1
│   ├── selected_indices.csv.xz   # Queried sample indices
│   └── full_auc_*.csv.xz         # Aggregated AUC metrics
├── 05_done_workload.csv          # 4.6M completed experiments
└── 01_workload.csv               # Full hyperparameter grid
```

---

## C. Research Starter Kit

### C1. Stopping Point Analysis

**Goal:** Detect when AL should stop and identify performance plateaus.

??? info "Required Inputs"
    | File | Location |
    |------|----------|
    | Per-cycle metrics | `<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` |
    | Selected indices | `<STRATEGY>/<DATASET>/selected_indices.csv.xz` |

**Run:**

```bash
# Generate learning curves
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan

# Export exemplary curves
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

**Outputs:** `_TS/weighted_f1-score.parquet`, `plots/single_learning_curve/*.parquet`

**Analysis ideas:**

- Detect plateau onset via derivative thresholds
- Compare stopping points across strategies
- Correlate early-iteration signals with final performance

---

### C2. Strategy Recommendation / Meta-Learning

**Goal:** Build a meta-model that recommends AL strategies based on dataset features.

??? info "Required Inputs"
    | File | Location |
    |------|----------|
    | Categorizations | `_/<CATEGORIZER>/<DATASET>.npz` |
    | AUC summaries | `<STRATEGY>/<DATASET>/full_auc_*.csv.xz` |

**Run:**

```bash
# Verify categorizations exist
ls ${OGAL_OUTPUT}/full_exp_jan/_/

# Load and join for meta-learning
python << 'EOF'
import pandas as pd
import os
OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")
print(ts.groupby(['EXP_DATASET', 'EXP_STRATEGY'])['metric_value'].mean().head())
EOF
```

**Analysis ideas:**

- Train meta-model to predict best strategy from dataset features
- Use early AUC (`first_5_*.csv.xz`) as cheap proxy
- Split by dataset clusters for leakage-safe evaluation

---

### C3. Dataset Similarity Clustering

Cluster datasets by AL behavior similarity.

```bash
python << 'EOF'
import numpy as np
from pathlib import Path
import os
OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")
cats = Path(f"{OGAL_OUTPUT}/full_exp_jan/_/REGION_DENSITY/")
print(f"Datasets available: {len(list(cats.glob('*.npz')))}")
EOF
```

→ See [Results Schema](reference/results_schema.md) for categorization details.

---

### C4. Trade-off Analysis

Compare accuracy vs. F1 vs. runtime across strategies.

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

→ See [Eva Scripts](reference/eva_scripts_catalog.md) for correlation analysis details.

---

### C5. Failure Mode Discovery

Identify when and why strategies fail.

```bash
python << 'EOF'
import pandas as pd
import os
OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")
failed = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_failed_workloads.csv")
print(failed.groupby('EXP_STRATEGY').size().sort_values(ascending=False).head(10))
EOF
```

→ Correlate failures with dataset properties (size, dimensionality, class imbalance).

---

## A. Produce Key Artifacts

Generate paper-style leaderboards and plots.

```bash
# Minimal evaluation chain
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Outputs:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

→ See [Eva Scripts Catalog](reference/eva_scripts_catalog.md) for complete script reference.

---

## B. Three Correlations (Paper)

| Correlation | Color | Command |
|-------------|-------|---------|
| Metric-based (Pearson) | Blue | `python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan` |
| Queried samples (Jaccard) | Green | `python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan` |
| Ranking invariance (Kendall) | Orange | `python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan` |

→ See [Correlations: Paper ↔ Code](reference/correlations_paper_to_code.md) for mathematical definitions.

---

## Reference

| Topic | Page |
|-------|------|
| Eva scripts catalog | [Eva Scripts](reference/eva_scripts_catalog.md) |
| Output file schemas | [Results Schema](reference/results_schema.md) |
| All AL strategies | [Strategy Catalog](reference/strategy_catalog.md) |
| Running experiments | [Runbook](reference/runbook.md) |
