# Data Model Reference

This document defines the canonical data model used throughout OGAL, including run identity, workload schema, result schemas, and the rationale for using integer enums.

---

## Run Identity (Primary Key)

Each experiment has a unique identity defined by the combination of hyperparameters. This is the primary key for all result data.

### Definition

**Run Identity = (EXP_DATASET, EXP_STRATEGY, EXP_LEARNER_MODEL, EXP_BATCH_SIZE, EXP_START_POINT, EXP_TRAIN_TEST_BUCKET_SIZE, EXP_RANDOM_SEED)**

Additionally, each run has a unique `EXP_UNIQUE_ID` integer assigned during workload creation.

(source: [`01_create_workload.py::_generate_exp_param_grid`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py#L40-L98), lines 40-98)

### Fields

| Field | Type | Description | Stability |
|-------|------|-------------|-----------|
| `EXP_UNIQUE_ID` | int | Auto-assigned row index (after shuffle) | Experiment-specific |
| `EXP_DATASET` | int | Dataset enum value | Stable across runs |
| `EXP_STRATEGY` | int | AL strategy enum value | Stable across runs |
| `EXP_LEARNER_MODEL` | int | Learner model enum value | Stable across runs |
| `EXP_BATCH_SIZE` | int | Query batch size | Config-defined |
| `EXP_START_POINT` | int | Initial labeled set index | Config-defined |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split bucket | Config-defined |
| `EXP_RANDOM_SEED` | int | Random seed | Config-defined |
| `EXP_NUM_QUERIES` | int | Number of AL cycles | Config-defined |

(source: [`misc/config.py::Config`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) class attributes, lines 65-82)

### Why `EXP_UNIQUE_ID`?

The composite key (7 fields) is unwieldy for joins. `EXP_UNIQUE_ID` provides:
- Single-column joins
- Faster lookups
- Shorter file paths
- Consistent ordering within an experiment

**Note:** `EXP_UNIQUE_ID` is only unique within a single experiment (`EXP_TITLE`). Different experiments may reuse the same IDs.

---

## Enum Definitions

OGAL uses integer enums for all categorical values. This provides:

| Benefit | Explanation |
|---------|-------------|
| **Stability** | Enum values never change, even if names are refactored |
| **Performance** | Integer comparisons are faster than string comparisons |
| **Join efficiency** | Integer keys produce smaller indexes |
| **No string drift** | Typos in config are caught immediately |
| **Serialization** | Integers serialize more compactly than strings |

### AL_STRATEGY Enum

**Source:** [`resources/data_types.py::AL_STRATEGY`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)

| ID | Name | Framework | Description |
|---:|------|-----------|-------------|
| 1 | `ALIPY_RANDOM` | ALiPy | Random sampling baseline |
| 2 | `ALIPY_UNCERTAINTY_LC` | ALiPy | Least Confident |
| 33 | `ALIPY_UNCERTAINTY_MM` | ALiPy | Max Margin |
| 34 | `ALIPY_UNCERTAINTY_ENTROPY` | ALiPy | Entropy-based |
| 9 | `LIBACT_UNCERTAINTY_LC` | libact | Least Confident |
| 12 | `LIBACT_QUIRE` | libact | QUIRE |
| 39 | `SMALLTEXT_LEASTCONFIDENCE` | small-text | Least Confident |
| 60 | `SKACTIVEML_QBC` | scikit-activeml | Query by Committee |
| ... | ... | ... | ... |

(source: [`resources/data_types.py::AL_STRATEGY`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py#L131-L208), lines 131-208)

### DATASET Enum

**Source:** [`datasets/__init__.py::DATASET`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py)

| ID | Name | Source |
|---:|------|--------|
| 1 | `Iris` | OpenML |
| 2 | `wine_origin` | OpenML |
| ... | ... | ... |

**Note:** Datasets are dynamically extended from `resources/local_datasets.yaml` if additional local datasets are defined.

(source: [`datasets/__init__.py::DATASET`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py); [`misc/config.py::Config._load_exp_yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), lines 358-367)

### LEARNER_MODEL Enum

**Source:** [`resources/data_types.py::LEARNER_MODEL`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)

| ID | Name | Class | Parameters |
|---:|------|-------|------------|
| 1 | `RF` | `RandomForestClassifier` | `n_jobs=cpu_count()` |
| 2 | `DT` | `DecisionTreeClassifier` | Default |
| 5 | `RBF_SVM` | `SVC` | `kernel="rbf", probability=True` |
| 8 | `MLP` | `MLPClassifier` | `hidden_layer_sizes=(100,)` |
| ... | ... | ... | ... |

(source: [`resources/data_types.py::LEARNER_MODEL`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py#L370-L384), lines 370-384; `learner_models_to_classes_mapping`, lines 389-467)

### COMPUTED_METRIC Enum

**Source:** [`resources/data_types.py::COMPUTED_METRIC`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)

| ID | Name | Description |
|---:|------|-------------|
| 1 | `STANDARD_AUC` | Area under learning curve |
| 2 | `DISTANCE_METRICS` | Batch/labeled/unlabeled distances |
| 3 | `MISMATCH_TRAIN_TEST` | Train/test distribution mismatch |
| 4 | `CLASS_DISTRIBUTIONS` | Per-cycle class balance |
| 5 | `METRIC_DROP` | Performance drops between cycles |
| 6 | `DATASET_CATEGORIZATION` | Per-sample characteristics |
| 7 | `TIMELAG_METRIC` | Time-lagged correlations |

(source: [`resources/data_types.py::COMPUTED_METRIC`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py#L491-L499), lines 491-499)

### SAMPLES_CATEGORIZER Enum

**Source:** [`resources/data_types.py::SAMPLES_CATEGORIZER`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)

| ID | Name | Description |
|---:|------|-------------|
| 1 | `COUNT_WRONG_CLASSIFICATIONS` | How often sample is misclassified |
| 2 | `SWITCHES_CLASS_OFTEN` | Prediction stability |
| 3 | `CLOSENESS_TO_DECISION_BOUNDARY` | Boundary proximity |
| 4 | `REGION_DENSITY` | Local density |
| 5 | `MELTING_POT_REGION` | Mixed-class region |
| 6 | `INCLUDED_IN_OPTIMAL_STRATEGY` | In oracle solution |
| 7 | `CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN` | Same-class kNN distance |
| 8 | `CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN` | Other-class kNN distance |
| 9 | `CLOSENESS_TO_CLUSTER_CENTER` | Cluster centroid distance |
| 10 | `IMPROVES_ACCURACY_BY` | Accuracy improvement if labeled |
| 11 | `AVERAGE_UNCERTAINTY` | Mean prediction uncertainty |
| 12 | `OUTLIERNESS` | Isolation forest score |
| 13 | `CLOSENESS_TO_SAMPLES_OF_SAME_CLASS` | Same-class distance |
| 14 | `CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS` | Other-class distance |

(source: [`resources/data_types.py::SAMPLES_CATEGORIZER`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py#L512-L528), lines 512-528)

---

## Workload Schema

The workload CSV (`01_workload.csv`) defines all experiments to run.

**Source:** [`01_create_workload.py::_generate_exp_param_grid`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Primary key (row index after shuffle) |
| `EXP_DATASET` | int | Dataset enum value |
| `EXP_STRATEGY` | int | Strategy enum value |
| `EXP_LEARNER_MODEL` | int | Learner model enum value |
| `EXP_BATCH_SIZE` | int | Samples queried per AL cycle |
| `EXP_RANDOM_SEED` | int | Random seed |
| `EXP_START_POINT` | int | Initial labeled set index |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split index |
| `EXP_NUM_QUERIES` | int | Number of AL cycles |

### Example

```csv
EXP_UNIQUE_ID,EXP_DATASET,EXP_STRATEGY,EXP_LEARNER_MODEL,EXP_BATCH_SIZE,EXP_RANDOM_SEED,EXP_START_POINT,EXP_TRAIN_TEST_BUCKET_SIZE,EXP_NUM_QUERIES
0,3,2,1,5,0,0,0,100
1,3,7,1,5,0,0,0,100
2,5,2,1,5,0,0,0,100
```

---

## Result Schemas

### Per-Cycle Metric File

**Location:** `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/<metric>.csv.xz`

**Source:** [`metrics/base_metric.py::Base_Metric.save_metrics`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/base_metric.py)

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Links to workload row |
| `0` | float | Metric value at AL cycle 0 |
| `1` | float | Metric value at AL cycle 1 |
| ... | ... | ... |
| `N` | float | Metric value at AL cycle N |

**Available metrics:**
- `accuracy.csv.xz` - Classification accuracy
- `weighted_f1-score.csv.xz` - Weighted F1 score
- `macro_f1-score.csv.xz` - Macro F1 score
- `weighted_precision.csv.xz` - Weighted precision
- `weighted_recall.csv.xz` - Weighted recall
- `selected_indices.csv.xz` - Queried sample indices (list per cell)
- `query_selection_time.csv.xz` - Query timing in seconds

(source: [`metrics/Standard_ML_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Standard_ML_Metrics.py), [`metrics/Selected_Indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Selected_Indices.py), [`metrics/Timing_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Timing_Metrics.py))

### Derived Metric File

**Location:** `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/<prefix>_<metric>.csv.xz`

**Source:** [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py)

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | int | Links to workload row |
| `value` | float | Aggregated metric value |

**Prefixes:**
- `full_auc_` - AUC over entire learning curve
- `ramp_up_auc_` - AUC during initial phase
- `plateau_auc_` - AUC during plateau phase
- `first_5_` - Mean of first 5 cycles
- `last_5_` - Mean of last 5 cycles
- `final_value_` - Last cycle value

(source: [`metrics/computed/STANDARD_AUC.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/STANDARD_AUC.py))

### Completion Tracking Files

**Done workload:** `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv`

Same schema as `01_workload.csv` - rows are appended on successful completion.

(source: [`misc/config.py::Config.OVERALL_DONE_WORKLOAD_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

**Failed workload:** `OUTPUT_PATH/<EXP_TITLE>/05_failed_workloads.csv`

| Column | Type | Description |
|--------|------|-------------|
| (All workload columns) | ... | Same as `01_workload.csv` |
| `error` | str | Exception type that caused failure |

(source: [`misc/config.py::Config.OVERALL_FAILED_WORKLOAD_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

**OOM workload:** `OUTPUT_PATH/<EXP_TITLE>/05_started_oom_workloads.csv`

Same schema as `01_workload.csv` - rows written before experiment starts, removed on success.

(source: [`misc/config.py::Config.OVERALL_STARTED_OOM_WORKLOAD_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

---

## Time Series Schema

Eva scripts create consolidated time series files for correlation analysis.

**Location:** `OUTPUT_PATH/<EXP_TITLE>/_TS/<metric>.parquet`

**Source:** [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py)

| Column | Type | Description |
|--------|------|-------------|
| `EXP_DATASET` | int | Dataset enum value |
| `EXP_STRATEGY` | int | Strategy enum value |
| `EXP_START_POINT` | int | Initial labeled set index |
| `EXP_BATCH_SIZE` | int | Query batch size |
| `EXP_LEARNER_MODEL` | int | Learner model enum value |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split index |
| `ix` | int | AL cycle index |
| `EXP_UNIQUE_ID_ix` | str | Composite key: `{EXP_UNIQUE_ID}_{ix}` |
| `metric_value` | float | Metric value at this cycle |

---

## Dataset Categorization Schema

Per-sample characteristics computed for each dataset.

**Location:** `OUTPUT_PATH/<EXP_TITLE>/_<CATEGORIZER>/<DATASET>.npz`

**Source:** [`metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py)

**Format:** NumPy compressed array with key `samples_categorization`

**Shape:** `(num_samples,)` or `(num_samples, num_features)` depending on categorizer

(source: [`metrics/computed/base_samples_categorizer.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py#L48-L67), lines 48-67)

---

## ID Mapping Tables

### Strategy Mapping

To convert between IDs and names:

```python
from resources.data_types import AL_STRATEGY

# ID to name
strategy_name = AL_STRATEGY(2).name  # "ALIPY_UNCERTAINTY_LC"

# Name to ID
strategy_id = AL_STRATEGY["ALIPY_UNCERTAINTY_LC"].value  # 2

# Get all strategies
for strategy in AL_STRATEGY:
    print(f"{strategy.value}: {strategy.name}")
```

(source: [`resources/data_types.py::AL_STRATEGY`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py))

### Dataset Mapping

```python
from datasets import DATASET

# ID to name
dataset_name = DATASET(3).name  # e.g., "Iris"

# Name to ID
dataset_id = DATASET["Iris"].value
```

(source: [`datasets/__init__.py::DATASET`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py))

### Learner Model Mapping

```python
from resources.data_types import LEARNER_MODEL, learner_models_to_classes_mapping

# Get the class for a model
model_class, params = learner_models_to_classes_mapping[LEARNER_MODEL.RF]
# model_class = RandomForestClassifier, params = {"n_jobs": cpu_count()}
```

(source: [`resources/data_types.py::learner_models_to_classes_mapping`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py#L389-L467), lines 389-467)

---

## Cross-References

- **[Architecture](architecture.md)**: System design and data flow
- **[Configuration](configuration.md)**: Config file reference
- **[Dataset Metadata](dataset_metadata.md)**: Auto-computed categorizations
- **[Results Format](results_format.md)**: Output file details
- **[Data Enrichment](data_enrichment.md)**: Adding new results
