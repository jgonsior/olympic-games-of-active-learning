# Analyze the Dataset

Use the published OPARA dataset (4.6M experiments) for your own research — without rerunning anything.

!!! success "What you can do"
    - **Start new research** using pre-computed results
    - **Produce paper-style artifacts** (leaderboards, heatmaps)
    - **Compute the three correlations** from the paper

---

## Setup

### 1. Get the Archived Data

**Canonical source:** [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

??? note "Convenience link (may change)"
    Direct download (not guaranteed stable):
    ```
    https://opara.zih.tu-dresden.de/xmlui/bitstream/handle/123456789/5678/full_exp_jan.zip
    ```

```bash
# Download (~several GB compressed)
wget <URL_FROM_DOI_LANDING_PAGE>

# Extract
unzip full_exp_jan.zip -d /path/to/results/full_exp_jan
```

### 2. Setup OGAL Environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning

conda create --name ogal --file conda-linux-64.lock
conda activate ogal
poetry install
```

### 3. Configure Paths

Define `OGAL_OUTPUT` once. All commands below reuse this variable.

```bash
export OGAL_OUTPUT=/path/to/results
cat > .server_access_credentials.cfg << EOF
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF
```

---

## C. Research Starter Kit

These analyses can be done **post-hoc** using existing OGAL outputs — no new experiments needed.

### C1. Stopping Point Analysis

**Goal:** Detect when AL should stop and identify performance plateaus.

**Research question:** When should AL stop? Can we detect plateaus automatically?

??? info "Required Inputs"

    | File | Location | Source |
    |------|----------|--------|
    | Per-cycle metrics | `<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` | `02_run_experiment.py` |
    | Selected indices | `<STRATEGY>/<DATASET>/selected_indices.csv.xz` | `02_run_experiment.py` |

**Run:**

```bash
# 1. Generate learning curves (creates _TS/*.parquet time series)
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan

# 2. Export exemplary curves
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

**You should see:**

| File | Description |
|------|-------------|
| `_TS/weighted_f1-score.parquet` | Time series for all experiments |
| `plots/single_learning_curve/weighted_f1-score.parquet` | Learning curve data |

!!! tip "Sanity check"
    If `_TS/weighted_f1-score.parquet` does not appear, see [Reference → Eva Scripts Catalog → learning_curve.py](reference/eva_scripts_catalog.md#learning_curvepy).

**Analysis ideas:**

- Detect plateau onset via derivative thresholds on exported curves
- Compare stopping points across strategies/datasets
- Correlate early-iteration signals with final performance

??? warning "Troubleshooting"
    - **Missing parquet files:** Run `learning_curve.py` first to create `_TS/` time series
    - **Memory errors:** Use `scripts/reduce_to_dense.py` to create smaller subsets

---

### C2. Strategy Recommendation / Meta-Learning

**Goal:** Build a meta-model that recommends AL strategies based on dataset features.

**Research question:** Can we recommend strategies based on dataset characteristics?

??? info "Required Inputs"

    | File | Location | Source |
    |------|----------|--------|
    | Dataset categorizations | `_/<CATEGORIZER>/<DATASET>.npz` | `03_calculate_dataset_categorizations.py` |
    | AUC summaries | `<STRATEGY>/<DATASET>/full_auc_*.csv.xz` | `04_calculate_advanced_metrics.py` |
    | Time series | `_TS/*.parquet` | `misc/helpers.py::create_fingerprint_joined_timeseries_csv_files` |

**Run:**

```bash
# 1. Verify categorizations exist (already in archive)
ls ${OGAL_OUTPUT}/full_exp_jan/_/

# 2. Load and join for meta-learning features
python << 'EOF'
import pandas as pd
from pathlib import Path
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load time series fingerprints
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")

# Join with dataset metadata for meta-learning
# Group by dataset to get dataset-level features
dataset_performance = ts.groupby(['EXP_DATASET', 'EXP_STRATEGY'])['metric_value'].mean()
print(dataset_performance.head())
EOF
```

**You should see:**

| Output | Description |
|--------|-------------|
| Console output | Dataset-strategy performance matrix |
| `_/` directory | Pre-computed categorizations in NPZ format |

!!! tip "Sanity check"
    If the `_/` directory is empty, run `03_calculate_dataset_categorizations.py`. See [Reference → Eva Scripts Catalog](reference/eva_scripts_catalog.md).

**Analysis ideas:**

- Train a meta-model to predict best strategy from dataset features
- Use early-iteration AUC (`first_5_*.csv.xz`) as cheap proxy for full run
- **Leakage-safe splits:** Split by dataset or dataset clusters

??? warning "Troubleshooting"
    - **Categorizations missing:** Archive includes pre-computed NPZ files in `_/` directory
    - **Memory issues:** Load one dataset at a time

---

### C3. Dataset Similarity Clustering

**Goal:** Identify which datasets behave similarly under AL.

**Research question:** Which datasets behave similarly under AL?

??? info "Required Inputs"

    | File | Location | Source |
    |------|----------|--------|
    | Categorizations | `_/<CATEGORIZER>/<DATASET>.npz` | `03_calculate_dataset_categorizations.py` |
    | Per-strategy counts | `<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz` | `metrics/computed/DATASET_CATEGORIZATION.py` |

**Run:**

```bash
# Load categorization vectors for clustering
python << 'EOF'
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load per-sample categorizations for each dataset
categorizations_dir = Path(f"{OGAL_OUTPUT}/full_exp_jan/_/REGION_DENSITY/")
datasets = {}
for npz_file in categorizations_dir.glob("*.npz"):
    data = np.load(npz_file)
    # Array key is typically 'arr_0' for np.savez default
    datasets[npz_file.stem] = data['arr_0']

# Compute dataset embeddings (e.g., mean/std of categorizations)
# Cluster datasets
print(f"Loaded {len(datasets)} datasets")
EOF
```

!!! note "Array key name"
    The array key in NPZ files is typically `arr_0` when saved with `np.savez` without explicit names. Verify by inspecting `data.files` if unsure.

**You should see:**

| Output | Description |
|--------|-------------|
| Console output | Number of loaded datasets |
| `datasets` dict | Per-sample categorization arrays |

!!! tip "Sanity check"
    If no NPZ files are found, verify the archive was extracted correctly and the `_/REGION_DENSITY/` directory exists.

**Analysis ideas:**

- Cluster datasets by hardness/density profiles
- Analyze strategy performance consistency within clusters
- Identify dataset families where specific strategies excel

---

### C4. Trade-off Analysis Across Metrics

**Goal:** Discover whether strategies that optimize one metric also optimize others.

**Research question:** Do strategies that optimize accuracy also optimize F1? What about runtime?

??? info "Required Inputs"

    | File | Location | Source |
    |------|----------|--------|
    | AUC metrics | `<STRATEGY>/<DATASET>/full_auc_accuracy.csv.xz` | `04_calculate_advanced_metrics.py` |
    | | `<STRATEGY>/<DATASET>/full_auc_weighted_f1-score.csv.xz` | |
    | Runtime | `<STRATEGY>/<DATASET>/query_selection_time.csv.xz` | `02_run_experiment.py` |

**Run:**

```bash
# 1. Compute metric correlations
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan

# 2. Compute AUC correlations
python -m eva_scripts.auc_metric_correlation --EXP_TITLE full_exp_jan

# 3. Analyze runtime
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

**You should see:**

| File | Description |
|------|-------------|
| `plots/basic_metrics/Standard Metrics.parquet` | Metric correlation matrix |
| `plots/AUC/auc_weighted_f1-score.parquet` | AUC aggregation correlations |
| `plots/runtime/query_selection_time.parquet` | Strategy runtime data |

!!! tip "Sanity check"
    If `plots/basic_metrics/` is empty, see [Reference → Eva Scripts Catalog → basic_metrics_correlation.py](reference/eva_scripts_catalog.md#basic_metrics_correlationpy).

**Analysis ideas:**

- Identify Pareto-optimal strategies (accuracy vs. runtime)
- Analyze when metric disagreements occur
- Find strategies robust across multiple metrics

---

### C5. Failure Mode Discovery

**Goal:** Identify when and why AL strategies fail.

**Research question:** When do strategies fail? Are there dataset characteristics that predict failure?

??? info "Required Inputs"

    | File | Location | Source |
    |------|----------|--------|
    | Failed workload | `05_failed_workloads.csv` | `02_run_experiment.py` |
    | OOM workload | `05_started_oom_workloads.csv` | `02_run_experiment.py` |
    | Dataset categorizations | `_/<CATEGORIZER>/<DATASET>.npz` | `03_calculate_dataset_categorizations.py` |

**Run:**

```bash
# Analyze failure patterns
python << 'EOF'
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

failed = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_failed_workloads.csv")

# Count failures by strategy
print("Failures by strategy:")
print(failed.groupby('EXP_STRATEGY').size().sort_values(ascending=False))

# Count failures by error type
print("\nFailures by error type:")
print(failed.groupby('error').size().sort_values(ascending=False))
EOF
```

**You should see:**

| Output | Description |
|--------|-------------|
| Console output | Failure counts grouped by strategy and error type |

!!! tip "Sanity check"
    If `05_failed_workloads.csv` does not exist, the archive may have zero failures (check `05_done_workload.csv` for completed runs).

**Analysis ideas:**

- Correlate failures with dataset properties (size, dimensionality, class imbalance)
- Identify strategy-dataset combinations that consistently fail
- Compare OOM patterns across strategies

---

## A. Produce Key Artifacts

Generate paper-style leaderboards and plots from the archived data.

### Minimal Evaluation Chain

**Goal:** Produce a publication-ready leaderboard heatmap.

**Inputs:**

| Input | Source |
|-------|--------|
| Done workload | `05_done_workload.csv` from archive |
| Per-cycle metrics | `<STRATEGY>/<DATASET>/*.csv.xz` from archive |

**Run:**

```bash
# Step 1: Basic metrics correlation
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan

# Step 2: Leaderboard rankings
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan

# Step 3: Final leaderboard heatmap
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan

# Step 4: (Optional) Publication-ready plots
python -m eva_scripts.redo_plots_for_paper --EXP_TITLE full_exp_jan
```

**You should see:**

| Artifact | Location | Description |
|----------|----------|-------------|
| **Leaderboard** | `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet` | Strategy rankings by dataset |
| **Learning curves** | `plots/single_learning_curve/*.parquet` | Figures 2-3 in paper |
| **Runtime chart** | `plots/runtime/query_selection_time.parquet` | Strategy computational costs |

### Checkpoints

| If this file is missing... | Run this script |
|---------------------------|-----------------|
| `_TS/*.parquet` | `python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan` |
| `plots/basic_metrics/*.parquet` | `python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan` |
| `plots/final_leaderboard/*.parquet` | `python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan` |

!!! tip "Sanity check"
    After each command, verify the output file exists. If missing, see [Eva Scripts Catalog](reference/eva_scripts_catalog.md) for prerequisites.

For the complete script catalog, see [Eva Scripts Catalog](reference/eva_scripts_catalog.md).

---

## B. Three Correlations (Paper Names)

The paper uses three correlation types with specific names. For detailed mathematical definitions, see [Correlations: Paper ↔ Code](reference/correlations_paper_to_code.md).

### Summary

| Paper Name | Measures | Heatmap Color | Core Function |
|------------|----------|---------------|---------------|
| **Metric-based** (Pearson) | Do two settings produce similar metric values? | Blue | `scipy.stats.pearsonr` |
| **Queried samples-based** (Jaccard) | Do two settings select the same samples? | Green | Set intersection/union |
| **Leaderboard ranking invariance** (Kendall τ-b) | Do two settings rank strategies the same way? | Orange | `scipy.stats.kendalltau` |

### Commands

**Goal:** Generate the three correlation heatmaps from the paper.

**Inputs:**

| Input | Source |
|-------|--------|
| Time series | `_TS/*.parquet` (run `learning_curve.py` first if missing) |
| Selected indices | `_TS/selected_indices.parquet` |

**Run:**

```bash
# 1. Metric-based correlation (Pearson) → Blue heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan

# 2. Queried samples-based correlation (Jaccard) → Green heatmaps
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan

# 3. Leaderboard ranking invariance (Kendall τ-b) → Orange heatmaps
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

**You should see:**

| Correlation | Output Location |
|-------------|-----------------|
| Metric-based (Pearson) | `plots/single_hyperparameter/*/single_hyper_*.parquet` |
| Queried samples (Jaccard) | `plots/single_hyperparameter/*/single_indice_*.parquet` |
| Ranking invariance (Kendall) | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` |

!!! tip "Sanity check"
    If `_TS/*.parquet` files are missing, first run `python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan`.

---

## Further Reading (Reference)

| Topic | Page |
|-------|------|
| Complete eva_scripts catalog | [Eva Scripts Catalog](reference/eva_scripts_catalog.md) |
| Mathematical correlation definitions | [Correlations: Paper ↔ Code](reference/correlations_paper_to_code.md) |
| Output file schemas | [Results Schema](reference/results_schema.md) |
| All AL strategies | [Strategy Catalog](reference/strategy_catalog.md) |
| Running experiments yourself | [Runbook](reference/runbook.md) |
