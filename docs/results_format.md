# Results Format

This document describes the output directory structure, file formats, and result schemas produced by OGAL experiments.

## Output Directory Structure

All experiment outputs are stored under `OUTPUT_PATH/<EXP_TITLE>/`:

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
        ├── retraining_time.csv
        └── y_pred_*.parquet
```

---

## Workload Files

### 01_workload.csv

The main workload file defining all experiment configurations.

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Unique identifier for this configuration |
| `EXP_DATASET` | int | Dataset enum value |
| `EXP_STRATEGY` | int | AL strategy enum value |
| `EXP_LEARNER_MODEL` | int | Learner model enum value |
| `EXP_BATCH_SIZE` | int | Query batch size |
| `EXP_RANDOM_SEED` | int | Random seed |
| `EXP_START_POINT` | int | Initial labeled set index |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split bucket |
| `EXP_NUM_QUERIES` | int | Number of AL iterations |

### 05_done_workload.csv

Tracks successfully completed experiments. Same schema as `01_workload.csv`.

### 05_failed_workloads.csv

Tracks failed experiments with error information.

| Column | Type | Description |
|--------|------|-------------|
| (All columns from workload) | | Same as 01_workload.csv |
| `error` | str | Exception type that caused failure |

Common error types:
- `<class 'sklearn.exceptions.ConvergenceWarning'>`: Model convergence issues
- `<class 'MemoryError'>`: Out of memory
- `<class 'TimeoutError'>`: Query selection exceeded time limit

### 05_started_oom_workloads.csv

Experiments that started but were killed by OOM killer. Same schema as `01_workload.csv`.

---

## Per-Cycle Metric Files

Located at `<STRATEGY>/<DATASET>/<metric>.csv`

### Schema

All per-cycle metric files share this structure:

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Links to workload row |
| `0` | float | Metric value at iteration 0 |
| `1` | float | Metric value at iteration 1 |
| ... | ... | ... |
| `N` | float | Metric value at iteration N |

Missing values (e.g., early stopping) are represented as `NaN`.

### Available Metrics

#### Standard ML Metrics

| File | Description | Range |
|------|-------------|-------|
| `accuracy.csv` | Classification accuracy | [0, 1] |
| `weighted_f1-score.csv` | Class-weighted F1 score | [0, 1] |
| `macro_f1-score.csv` | Macro-averaged F1 score | [0, 1] |
| `weighted_precision.csv` | Class-weighted precision | [0, 1] |
| `weighted_recall.csv` | Class-weighted recall | [0, 1] |
| `macro_precision.csv` | Macro-averaged precision | [0, 1] |
| `macro_recall.csv` | Macro-averaged recall | [0, 1] |

#### Selection Metrics

| File | Description |
|------|-------------|
| `selected_indices.csv` | Sample indices selected per iteration |

Format: Each cell contains a list of selected indices (as string representation).

#### Timing Metrics

| File | Description | Unit |
|------|-------------|------|
| `query_selection_time.csv` | Time to select query samples | seconds |
| `retraining_time.csv` | Time to retrain the model | seconds |

### Prediction Files

| File | Format | Description |
|------|--------|-------------|
| `y_pred_train_<iteration>.parquet` | Parquet | Train set predictions per cycle |
| `y_pred_test_<iteration>.parquet` | Parquet | Test set predictions per cycle |

---

## Derived Metrics

Computed by `04_calculate_advanced_metrics.py`.

### AUC Metrics

Area-under-curve computations for learning curves:

| Metric | Description |
|--------|-------------|
| `full_auc_<metric>` | AUC over entire learning curve |
| `ramp_up_auc_<metric>` | AUC during initial "ramp-up" phase |
| `plateau_auc_<metric>` | AUC during plateau phase |
| `first_5_<metric>` | Average of first 5 iterations |
| `last_5_<metric>` | Average of last 5 iterations |
| `final_value_<metric>` | Final iteration value |

### Distance Metrics

Sample-level distance analysis:

| Metric | Description |
|--------|-------------|
| `DISTANCE_METRICS` | Pairwise distance statistics |
| `MISMATCH_TRAIN_TEST` | Train/test distribution divergence |

### Dataset Categorization

Per-sample characteristics computed by `03_calculate_dataset_categorizations.py`:

| Categorizer | Description |
|-------------|-------------|
| `COUNT_WRONG_CLASSIFICATIONS` | Misclassification frequency |
| `SWITCHES_CLASS_OFTEN` | Prediction instability |
| `CLOSENESS_TO_DECISION_BOUNDARY` | Margin from boundary |
| `REGION_DENSITY` | Local sample density |
| `MELTING_POT_REGION` | Mixed-class neighborhood |
| `OUTLIERNESS` | Outlier score |
| `AVERAGE_UNCERTAINTY` | Mean prediction uncertainty |

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
metrics = pd.read_csv("OUTPUT_PATH/test/metrics/some_metric.csv.xz")

# Load Parquet
predictions = pd.read_parquet("OUTPUT_PATH/test/ALIPY_RANDOM/Iris/y_pred_train_10.parquet")
```

---

## Mapping Scripts to Outputs

| Script | Primary Outputs |
|--------|-----------------|
| `00_download_datasets.py` | `DATASETS_PATH/*.csv`, `*_split.csv` |
| `01_create_workload.py` | `01_workload.csv`, `00_config.yaml`, SLURM files |
| `02_run_experiment.py` | `<STRATEGY>/<DATASET>/*.csv`, `05_*.csv` |
| `03_calculate_dataset_categorizations.py` | `workloads/DATASET_CATEGORIZATIONS/` |
| `04_calculate_advanced_metrics.py` | `workloads/advanced_metrics/`, AUC files |
| `05_analyze_partially_run_workload.py` | Analysis statistics |
| `07b_create_results_without_flask.py` | `plots/`, HTML reports |

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
│       │── # Aggregated metrics (computed by 04_calculate_advanced_metrics.py)
│       ├── full_auc_accuracy.csv.xz
│       ├── first_5_accuracy.csv.xz
│       ├── last_5_accuracy.csv.xz
│       ├── final_value_accuracy.csv.xz
│       ├── full_auc_weighted_f1-score.csv.xz
│       ├── full_auc_macro_f1-score.csv.xz
│       │
│       │── # Dataset categorizations (computed by 03_calculate_dataset_categorizations.py)
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
│       │── # Time-lag variants (for correlation analysis)
│       ├── accuracy_time_lag.csv.xz
│       ├── *_time_lag.csv.xz           # Time-lagged versions of all metrics
│       │
│       │── # Distance metrics
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

```bash
# Download from OPARA (manual download required due to authentication)
# Then extract:
unzip full_exp_jan.zip
```

#### 2. Load into OGAL for Analysis

Configure your `.server_access_credentials.cfg` to point to the extracted data:

```ini
[LOCAL]
LOCAL_DATASETS_PATH = /path/to/datasets/
LOCAL_OUTPUT_PATH = /path/to/full_exp_jan/
```

#### 3. Run Evaluation Scripts

The `eva_scripts/` can directly analyze the archived data:

```python
# Set experiment title to match archive
# python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan

# The archive structure matches OGAL's output format exactly
import pandas as pd

# Load completed workload
done = pd.read_csv("full_exp_jan/05_done_workload.csv")
print(f"Total completed experiments: {len(done)}")

# Load metrics for a specific strategy/dataset combination
accuracy = pd.read_csv(
    "full_exp_jan/ALIPY_CORESET_GREEDY/Bioresponse/accuracy.csv.xz"
)
print(f"Experiments for this combination: {len(accuracy)}")
```

#### 4. Reproduce Paper Figures

The evaluation scripts can regenerate paper figures from the archived data:

```bash
# After setting OUTPUT_PATH to point to the extracted archive:
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

### File Size Considerations

The complete archive is several terabytes when uncompressed. For initial exploration, you can:

1. **Extract selectively**: Only extract specific strategies or datasets
2. **Use dense workload**: The dense subset contains ~1.1M complete hyperparameter combinations
3. **Start with metrics only**: Skip `y_pred_*.parquet` files (prediction arrays) if not needed
