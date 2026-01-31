# Results Format

This document describes the output directory structure, file formats, and result schemas produced by OGAL experiments. All paths and schemas are verified against source code.

!!! tip "Related Documentation"
    - **[Eva Scripts](eva_scripts_catalog.md)**: Detailed catalog of evaluation scripts and their I/O
    - **[Data Enrichment](data_enrichment.md)**: Protocol for adding new results
    - **[Evaluation Pipeline](../analyze_dataset.md)**: How raw outputs become final figures

---

## Results Overview

OGAL results fall into two categories:

| Category | Source | Examples | Typical Location |
|----------|--------|----------|------------------|
| **Raw Experiment Outputs** | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) | `accuracy.csv`, `selected_indices.csv`, `05_done_workload.csv` | `<STRATEGY>/<DATASET>/` |
| **Derived Artifacts** | `03_*.py`, `04_*.py`, [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) | `full_auc_*.csv.xz`, `_TS/*.parquet`, `plots/*.parquet` | `<STRATEGY>/<DATASET>/`, `_TS/`, [`plots/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/plots) |

## Output Directory Structure

All experiment outputs are stored under `OUTPUT_PATH/<EXP_TITLE>/`:

(source: [`misc/config.py::Config._pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py#L224), line 224)

```
OUTPUT_PATH/<EXP_TITLE>/
├── 00_config.yaml                    # Saved experiment configuration
├── 01_workload.csv                   # Experiment workload definition
├── 01_non_hpc_workload.csv          # Strategies not suitable for HPC (optional)
├── 02_slurm.slurm                   # Generated SLURM job script
├── 02b_run_bash_parallel.py         # Local parallel runner script
├── 05_done_workload.csv             # Completed experiments
├── 05_failed_workloads.csv          # Failed experiments with errors
├── 05_started_oom_workloads.csv     # OOM-killed experiments
├── 06_dense_workload.csv            # Reduced workload (after cleanup)
├── 07_missing_exp_ids.csv           # Missing experiment IDs
├── 07_broken_csv_file_found.csv     # Corrupted result files
├── metrics/                          # Per-experiment-ID metric files
├── workloads/                        # Post-processing workloads
│   ├── DATASET_CATEGORIZATIONS/
│   └── advanced_metrics/
├── plots/                            # Generated visualizations
├── _TS/                              # Time series correlation data
└── <STRATEGY_NAME>/                  # Per-strategy results
    └── <DATASET_NAME>/               # Per-dataset results
        ├── accuracy.csv
        ├── weighted_f1-score.csv
        ├── macro_f1-score.csv
        ├── weighted_precision.csv
        ├── weighted_recall.csv
        ├── macro_precision.csv
        ├── macro_recall.csv
        ├── selected_indices.csv
        ├── query_selection_time.csv
        ├── learner_training_time.csv
        └── y_pred_*.csv.xz.parquet
```

---

## Primary Result Files Schema

### 01_workload.csv

The main workload file defining all experiment configurations.

(source: [`01_create_workload.py::_generate_exp_param_grid`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py#L40-L98), lines 40-98)

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `EXP_UNIQUE_ID` | int | Unique identifier (0-indexed row after shuffle) | Line 96 |
| `EXP_DATASET` | int | Dataset enum value | [`resources/data_types.py::DATASET`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) |
| `EXP_STRATEGY` | int | AL strategy enum value | [`resources/data_types.py::AL_STRATEGY`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) |
| `EXP_LEARNER_MODEL` | int | Learner model enum value | [`resources/data_types.py::LEARNER_MODEL`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) |
| `EXP_BATCH_SIZE` | int | Query batch size | [`misc/config.py::Config.EXP_BATCH_SIZE`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `EXP_RANDOM_SEED` | int | Random seed | [`misc/config.py::Config.EXP_RANDOM_SEED`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `EXP_START_POINT` | int | Initial labeled set index | [`misc/config.py::Config.EXP_START_POINT`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split bucket | [`misc/config.py::Config.EXP_TRAIN_TEST_BUCKET_SIZE`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `EXP_NUM_QUERIES` | int | Number of AL iterations | [`misc/config.py::Config.EXP_NUM_QUERIES`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |

### 05_done_workload.csv

Tracks successfully completed experiments. Same schema as `01_workload.csv`.

(source: [`framework_runners/base_runner.py::AL_Experiment.run_experiment`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py#L224-L229), lines 224-229)

### 05_failed_workloads.csv

Tracks failed experiments with error information.

(source: [`framework_runners/base_runner.py::AL_Experiment.run_experiment`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py#L210-L217), lines 210-217)

| Column | Type | Description |
|--------|------|-------------|
| (All columns from workload) | | Same as 01_workload.csv |
| `error` | str | Exception type that caused failure |

**Common error types:**

| Error | Meaning |
|-------|---------|
| `<class 'sklearn.exceptions.ConvergenceWarning'>` | Model convergence issues |
| `<class 'MemoryError'>` | Out of memory |
| `<class 'TimeoutError'>` | Query selection exceeded time limit |
| `<class 'OSError'>` | I/O or file system error |

### 05_started_oom_workloads.csv

Experiments that started but were presumed killed by OOM. Written **before** experiment runs, removed from tracking if it completes successfully.

(source: [`framework_runners/base_runner.py::AL_Experiment.run_experiment`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py#L120-L126), lines 120-126)

Same schema as `01_workload.csv` (no error column).

---

## Minimal Example: How Keys Tie Together

This section shows a concrete example of how `EXP_UNIQUE_ID` links data across files.

### Example Workload Row

From `05_done_workload.csv`:

| EXP_UNIQUE_ID | EXP_DATASET | EXP_STRATEGY | EXP_LEARNER_MODEL | EXP_BATCH_SIZE | EXP_RANDOM_SEED | EXP_START_POINT | EXP_TRAIN_TEST_BUCKET_SIZE | EXP_NUM_QUERIES |
|---------------|-------------|--------------|-------------------|----------------|-----------------|-----------------|---------------------------|-----------------|
| 12345 | 3 | 7 | 1 | 5 | 0 | 0 | 0 | 100 |

**Interpretation:**
- `EXP_UNIQUE_ID=12345`: Unique identifier for this experiment run
- `EXP_DATASET=3`: Dataset enum value (e.g., `DATASET.Iris.value = 3`)
- `EXP_STRATEGY=7`: Strategy enum value (e.g., `AL_STRATEGY.ALIPY_RANDOM.value = 7`)
- `EXP_LEARNER_MODEL=1`: Model enum value (e.g., `LEARNER_MODEL.RF.value = 1`)
- `EXP_BATCH_SIZE=5`: Query 5 samples per AL cycle
- `EXP_START_POINT=0`: Use first pre-generated initial labeled set
- `EXP_TRAIN_TEST_BUCKET_SIZE=0`: Use first train/test split

### Corresponding Metric File Row

From `ALIPY_RANDOM/Iris/accuracy.csv.xz`:

| EXP_UNIQUE_ID | 0 | 1 | 2 | 3 | ... | 99 |
|---------------|-----|-----|-----|-----|-----|------|
| 12345 | 0.72 | 0.78 | 0.82 | 0.85 | ... | 0.95 |

**Interpretation:**
- Same `EXP_UNIQUE_ID=12345` links this row to the workload
- Columns `0`, `1`, `2`, ... represent AL cycle indices
- Values are the metric (accuracy) at each cycle

### Corresponding Derived Metric

From `ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz`:

| EXP_UNIQUE_ID | value |
|---------------|-------|
| 12345 | 0.87 |

**Interpretation:**
- `full_auc_accuracy` = Area under the accuracy learning curve
- Computed by [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) from the per-cycle values

### Corresponding Time Series Entry

From `_TS/full_auc_accuracy.parquet`:

| EXP_DATASET | EXP_STRATEGY | EXP_START_POINT | EXP_BATCH_SIZE | EXP_LEARNER_MODEL | EXP_TRAIN_TEST_BUCKET_SIZE | ix | EXP_UNIQUE_ID_ix | metric_value |
|-------------|--------------|-----------------|----------------|-------------------|---------------------------|-----|------------------|--------------|
| 3 | 7 | 0 | 5 | 1 | 0 | 0 | 12345_0 | 0.87 |

**Interpretation:**
- Time series format used by eva_scripts for correlation analysis
- `EXP_UNIQUE_ID_ix` combines experiment ID with cycle index for unique identification

(source: [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py#L200-L267), lines 200-267)

---

## Part 1: Raw Experiment Outputs

These files are produced directly by [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) during experiment execution.

### Per-Cycle Metric Files Schema

Located at `<STRATEGY>/<DATASET>/<metric>.csv`

(source: [`metrics/base_metric.py::Base_Metric.save_metrics`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/base_metric.py))

### Common Schema for All Per-Cycle Metrics

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Links to workload row |
| `0` | float | Metric value at iteration 0 (initial) |
| `1` | float | Metric value at iteration 1 |
| ... | ... | ... |
| `N` | float | Metric value at iteration N |

Missing values (e.g., early stopping) are represented as empty cells or `NaN`.

### Standard ML Metrics

(source: [`metrics/Standard_ML_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Standard_ML_Metrics.py))

| File | Description | Range |
|------|-------------|-------|
| `accuracy.csv` | Classification accuracy | [0, 1] |
| `weighted_f1-score.csv` | Class-weighted F1 score | [0, 1] |
| `macro_f1-score.csv` | Macro-averaged F1 score | [0, 1] |
| `weighted_precision.csv` | Class-weighted precision | [0, 1] |
| `weighted_recall.csv` | Class-weighted recall | [0, 1] |
| `macro_precision.csv` | Macro-averaged precision | [0, 1] |
| `macro_recall.csv` | Macro-averaged recall | [0, 1] |

### Selection Metrics

(source: [`metrics/Selected_Indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Selected_Indices.py))

| File | Description |
|------|-------------|
| `selected_indices.csv` | Sample indices selected per iteration |

**Format:** Each cell contains a Python list representation of selected indices (e.g., `[5, 10, 15]`).

### Timing Metrics

(source: [`metrics/Timing_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Timing_Metrics.py))

| File | Description | Unit |
|------|-------------|------|
| `query_selection_time.csv` | Time to select query samples | seconds |
| `learner_training_time.csv` | Time to retrain the model | seconds |

### Prediction Files

(source: [`metrics/Predicted_Samples.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Predicted_Samples.py))

| File | Format | Description |
|------|--------|-------------|
| `y_pred_train.csv.xz.parquet` | Parquet | Train set predictions per cycle |
| `y_pred_test.csv.xz.parquet` | Parquet | Test set predictions per cycle |

---

## Part 2: Derived Artifacts (Post-Processing + Eva Scripts)

These files are computed from raw outputs by post-processing scripts and eva_scripts.

### Derived Metrics Schema

Computed by [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py).

(source: `metrics/computed/` directory)

### AUC Metrics

| File Pattern | Description | Computation |
|--------------|-------------|-------------|
| `full_auc_<metric>.csv.xz` | AUC over entire learning curve | `np.trapz(values)` |
| `ramp_up_auc_<metric>.csv.xz` | AUC during initial "ramp-up" phase | First N iterations |
| `plateau_auc_<metric>.csv.xz` | AUC during plateau phase | Later iterations |
| `first_5_<metric>.csv.xz` | Average of first 5 iterations | `mean(values[:5])` |
| `last_5_<metric>.csv.xz` | Average of last 5 iterations | `mean(values[-5:])` |
| `final_value_<metric>.csv.xz` | Final iteration value | `values[-1]` |

**Schema:** Same as per-cycle metrics but with a single value column.

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Links to workload row |
| `value` | float | Computed aggregated metric value |

### Distance Metrics

(source: [`metrics/computed/DISTANCE_METRICS.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DISTANCE_METRICS.py))

| File | Description |
|------|-------------|
| `avg_dist_batch.csv.xz` | Average pairwise distance within batch |
| `avg_dist_labeled.csv.xz` | Average distance to labeled samples |
| `avg_dist_unlabeled.csv.xz` | Average distance to unlabeled samples |

### Dataset Categorization Files

(source: [`metrics/computed/base_samples_categorizer.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py))

Per-sample characteristics computed by [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py):

| File | Description |
|------|-------------|
| `COUNT_WRONG_CLASSIFICATIONS.csv.xz` | How often sample is misclassified |
| `SWITCHES_CLASS_OFTEN.csv.xz` | Prediction instability count |
| `CLOSENESS_TO_DECISION_BOUNDARY.csv.xz` | Distance to decision boundary |
| `REGION_DENSITY.csv.xz` | Local sample density |
| `MELTING_POT_REGION.csv.xz` | Mixed-class neighborhood indicator |
| `OUTLIERNESS.csv.xz` | Outlier score |
| `AVERAGE_UNCERTAINTY.csv.xz` | Mean prediction uncertainty |

**Schema:** Per-sample values indexed by sample ID.

---

## File Formats

### CSV Files

- Delimiter: `,`
- Header: First row
- Compression: `.csv.xz` (LZMA) for large files

### Parquet Files

Used for large prediction arrays:

- Column-oriented format
- Efficient compression
- Schema preserved

### Loading Examples

```python
import pandas as pd

# Load workload
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")

# Load metric results
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Load compressed CSV
metrics = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz")

# Load Parquet predictions
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train.csv.xz.parquet")
```

---

## Aggregation Conventions

### By Random Seed

When aggregating across seeds (if multiple `EXP_GRID_RANDOM_SEED` values):

```python
# Group by all hyperparameters except seed
groupby_cols = ['EXP_DATASET', 'EXP_STRATEGY', 'EXP_LEARNER_MODEL', 
                'EXP_BATCH_SIZE', 'EXP_START_POINT', 'EXP_TRAIN_TEST_BUCKET_SIZE']
aggregated = df.groupby(groupby_cols)['final_accuracy'].agg(['mean', 'std'])
```

### By Dataset

(source: [`eva_scripts/final_leaderboard.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/final_leaderboard.py))

Leaderboard rankings are computed per-dataset, then aggregated:

```python
# Rank strategies within each dataset
df['rank'] = df.groupby('EXP_DATASET')['metric'].rank(ascending=False)
# Aggregate ranks across datasets
final_rank = df.groupby('EXP_STRATEGY')['rank'].mean()
```

### By Train/Test Split

Split bucket indices (0-4 by default) represent different random data partitions:

(source: [`misc/config.py::Config.EXP_GRID_TRAIN_TEST_BUCKET_SIZE`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

```python
# Aggregate across splits
aggregated = df.groupby(['EXP_DATASET', 'EXP_STRATEGY', 'EXP_LEARNER_MODEL'])['metric'].mean()
```

---

## Mapping Scripts to Outputs

| Script | Primary Outputs | Code Pointer |
|--------|-----------------|--------------|
| [`00_download_datasets.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/00_download_datasets.py) | `DATASETS_PATH/*.csv`, `*_split.csv` | [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py) |
| [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) | `01_workload.csv`, `00_config.yaml`, SLURM files | [`01_create_workload.py::create_workload`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) |
| [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) | `<STRATEGY>/<DATASET>/*.csv`, `05_*.csv` | [`framework_runners/base_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py) |
| [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py) | `<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz` | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py) |
| [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) | `<STRATEGY>/<DATASET>/full_auc_*.csv.xz` | [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) |
| [`05_analyze_partially_run_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/05_analyze_partially_run_workload.py) | Console output (analysis) | [`05_analyze_partially_run_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/05_analyze_partially_run_workload.py) |
| [`07b_create_results_without_flask.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/07b_create_results_without_flask.py) | [`plots/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/plots), HTML reports | [`07b_create_results_without_flask.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/07b_create_results_without_flask.py) |

---

## Aggregating Results

### Join Workload with Metrics

```python
import pandas as pd
from pathlib import Path

# Load completed workload
done = pd.read_csv("OUTPUT_PATH/test/05_done_workload.csv")

# Load a metric file
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Join on EXP_UNIQUE_ID
merged = done.merge(accuracy, on="EXP_UNIQUE_ID")
```

### Compute Summary Statistics

```python
# Get final accuracy for each experiment
accuracy_cols = [str(i) for i in range(100)]  # Assuming 100 iterations
merged["final_accuracy"] = merged[accuracy_cols].iloc[:, -1]

# Group by strategy
summary = merged.groupby("EXP_STRATEGY")["final_accuracy"].agg(["mean", "std"])
```

---

## Data Integrity

### Checking for Missing Results

```python
# Compare expected vs completed
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")
done = pd.read_csv("OUTPUT_PATH/test/05_done_workload.csv")

expected_ids = set(workload["EXP_UNIQUE_ID"])
completed_ids = set(done["EXP_UNIQUE_ID"])
missing_ids = expected_ids - completed_ids

print(f"Missing: {len(missing_ids)} / {len(expected_ids)}")
```

### Validating Metric Files

```python
from pathlib import Path
import pandas as pd

results_dir = Path("OUTPUT_PATH/test")
broken_files = []

for csv_file in results_dir.glob("**/*.csv"):
    try:
        df = pd.read_csv(csv_file)
        if "EXP_UNIQUE_ID" not in df.columns:
            broken_files.append(csv_file)
    except Exception as e:
        broken_files.append((csv_file, str(e)))
```

---

## OPARA Archived Results

The complete experiment results from the paper are archived at **[DOI: 10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)**.

### Archive Structure

The archive contains a ZIP file (`full_exp_jan.zip`) with the following structure:

```
full_exp_jan/
├── <STRATEGY_NAME>/                    # e.g., ALIPY_BMDR/, ALIPY_CORESET_GREEDY/
│   └── <DATASET_NAME>/                 # e.g., Bioresponse/, Iris/
│       ├── accuracy.csv.xz             # Per-cycle accuracy
│       ├── weighted_f1-score.csv.xz    # Per-cycle weighted F1
│       ├── macro_f1-score.csv.xz       # Per-cycle macro F1
│       ├── query_selection_time.csv.xz # Query selection timing
│       ├── learner_training_time.csv.xz # Model training timing
│       ├── y_pred_train.csv.xz.parquet # Train predictions
│       ├── y_pred_test.csv.xz.parquet  # Test predictions
│       ├── selected_indices.csv.xz     # Selected sample indices
│       │
│       ├── # Aggregated metrics (from 04_calculate_advanced_metrics.py)
│       ├── full_auc_accuracy.csv.xz
│       ├── first_5_accuracy.csv.xz
│       ├── last_5_accuracy.csv.xz
│       ├── final_value_accuracy.csv.xz
│       ├── full_auc_weighted_f1-score.csv.xz
│       ├── full_auc_macro_f1-score.csv.xz
│       │
│       ├── # Dataset categorizations (from 03_calculate_dataset_categorizations.py)
│       ├── AVERAGE_UNCERTAINTY.csv.xz
│       ├── CLOSENESS_TO_CLUSTER_CENTER.csv.xz
│       ├── CLOSENESS_TO_DECISION_BOUNDARY.csv.xz
│       ├── CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS.csv.xz
│       ├── CLOSENESS_TO_SAMPLES_OF_SAME_CLASS.csv.xz
│       ├── COUNT_WRONG_CLASSIFICATIONS.csv.xz
│       ├── IMPROVES_ACCURACY_BY.csv.xz
│       ├── INCLUDED_IN_OPTIMAL_STRATEGY.csv.xz
│       ├── MELTING_POT_REGION.csv.xz
│       ├── OUTLIERNESS.csv.xz
│       ├── REGION_DENSITY.csv.xz
│       ├── SWITCHES_CLASS_OFTEN.csv.xz
│       │
│       ├── # Time-lag variants (for correlation analysis)
│       ├── accuracy_time_lag.csv.xz
│       ├── *_time_lag.csv.xz           # Time-lagged versions of all metrics
│       │
│       ├── # Distance metrics
│       ├── avg_dist_batch.csv.xz
│       ├── avg_dist_labeled.csv.xz
│       └── avg_dist_unlabeled.csv.xz
├── 05_done_workload.csv                # Completed experiment configurations
├── 05_failed_workloads.csv             # Failed experiments
└── 01_workload.csv                     # Full workload definition
```

### Strategies in Archive

The archive contains results for 28 AL strategies from 5 frameworks:

| Framework | Strategies |
|-----------|------------|
| **ALiPy** | ALIPY_BMDR, ALIPY_CORESET_GREEDY, ALIPY_DENSITY_WEIGHTED, ALIPY_GRAPH_DENSITY, ALIPY_HIERARCHICAL, ALIPY_LAL, ALIPY_QBC, ALIPY_QUERY_BY_COMMITTEE, ALIPY_RANDOM, ALIPY_UNCERTAINTY_* |
| **libact** | LIBACT_* strategies |
| **scikit-activeml** | SKACTIVEML_* strategies |
| **small-text** | SMALL_TEXT_* strategies |
| **playground** | PLAYGROUND_* strategies |

### Datasets in Archive

The archive includes results for 92 datasets covering:

- **Binary classification**: 60 datasets
- **Multi-class classification**: 32 datasets (up to 31 classes)
- **Sample sizes**: 100 to 20,000 samples
- **Feature dimensions**: 2 to 1,776 features

### Using Archived Results

#### 1. Download and Extract

```python
import pandas as pd

# Load workload
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")

# Load metric results
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Load compressed CSV
metrics = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz")

# Load Parquet predictions
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train.csv.xz.parquet")
```000000

#### 2. Load into OGAL for Analysis

Configure your `.server_access_credentials.cfg` to point to the extracted data:

```python
import pandas as pd

# Load workload
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")

# Load metric results
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Load compressed CSV
metrics = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz")

# Load Parquet predictions
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train.csv.xz.parquet")
```111111

#### 3. Run Evaluation Scripts

The [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) can directly analyze the archived data:

```python
import pandas as pd

# Load workload
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")

# Load metric results
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Load compressed CSV
metrics = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz")

# Load Parquet predictions
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train.csv.xz.parquet")
```222222

#### 4. Reproduce Paper Figures

The evaluation scripts can regenerate paper figures from the archived data:

```python
import pandas as pd

# Load workload
workload = pd.read_csv("OUTPUT_PATH/test/01_workload.csv")

# Load metric results
accuracy = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/accuracy.csv")

# Load compressed CSV
metrics = pd.read_csv("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz")

# Load Parquet predictions
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train.csv.xz.parquet")
```333333

### File Size Considerations

The complete archive is several terabytes when uncompressed. For initial exploration, you can:

1. **Extract selectively**: Only extract specific strategies or datasets
2. **Use dense workload**: The dense subset contains ~1.1M complete hyperparameter combinations
3. **Start with metrics only**: Skip `y_pred_*.parquet` files (prediction arrays) if not needed
