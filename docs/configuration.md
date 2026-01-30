# Configuration Guide

OGAL uses a shared configuration system that controls all aspects of the experimental pipeline. This document explains the configuration files, their structure, and how they are loaded by each script.

## Configuration Files Overview

| File | Purpose | Format |
|------|---------|--------|
| `.server_access_credentials.cfg` | Machine-specific paths (local vs HPC) | INI |
| `resources/exp_config.yaml` | Experiment grid definitions | YAML |
| `OUTPUT_PATH/<EXP_TITLE>/00_config.yaml` | Saved experiment configuration | YAML |

## Server Access Credentials

**File:** `.server_access_credentials.cfg` (gitignored)

This file defines paths for both local development and HPC cluster environments. The `RUNNING_ENVIRONMENT` setting (default: `local`) determines which section is active.

### Structure

```ini
[HPC]
SSH_LOGIN=user@hpc_server
WS_PATH=/path/to/workspace
DATASETS_PATH=/path/to/datasets
OUTPUT_PATH=/path/to/results
SLURM_MAIL=your.email@example.org
SLURM_PROJECT="your_project_name"
CODE_PATH=/path/to/code
PYTHON_PATH=/path/to/python

[LOCAL]
DATASETS_PATH=/home/user/al_survey/datasets
CODE_PATH=/home/user/al_survey/code
OUTPUT_PATH=/home/user/al_survey/exp_results
```

### Field Reference

#### HPC Section

| Field | Description |
|-------|-------------|
| `SSH_LOGIN` | SSH connection string for HPC access |
| `WS_PATH` | HPC workspace base path |
| `DATASETS_PATH` | Path to datasets on HPC |
| `OUTPUT_PATH` | Path for experiment outputs on HPC |
| `SLURM_MAIL` | Email for SLURM notifications |
| `SLURM_PROJECT` | SLURM project/account name |
| `CODE_PATH` | Path to code repository on HPC |
| `PYTHON_PATH` | Path to Python interpreter on HPC |

#### LOCAL Section

| Field | Description |
|-------|-------------|
| `DATASETS_PATH` | Local path to datasets |
| `CODE_PATH` | Local path to code repository |
| `OUTPUT_PATH` | Local path for experiment outputs |

### How Scripts Load This File

The `Config` class in `misc/config.py` loads this file via `_load_server_setup_from_file()`:

1. Parses both `[HPC]` and `[LOCAL]` sections
2. Prefixes each key with section name (e.g., `HPC_DATASETS_PATH`, `LOCAL_DATASETS_PATH`)
3. Sets attributes on the Config instance
4. The `_pathes_magic()` method then selects the appropriate paths based on `RUNNING_ENVIRONMENT`

---

## Experiment Configuration

**File:** `resources/exp_config.yaml`

This file defines named experiment configurations as hyperparameter grids.

### Structure

```yaml
experiment_name:
  EXP_GRID_DATASET: [...]
  EXP_GRID_STRATEGY: [...]
  EXP_GRID_LEARNER_MODEL: [...]
  EXP_GRID_BATCH_SIZE: [...]
  EXP_GRID_RANDOM_SEED: [...]
  EXP_GRID_START_POINT: [...]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [...]
  EXP_GRID_NUM_QUERIES: [...]
  METRICS: [...]
```

### Key Fields

#### Grid Parameters (EXP_GRID_*)

These define the Cartesian product for the experiment workload:

| Field | Type | Description |
|-------|------|-------------|
| `EXP_GRID_DATASET` | List | Dataset identifiers (names or IDs) |
| `EXP_GRID_STRATEGY` | List | AL strategy names |
| `EXP_GRID_LEARNER_MODEL` | List | Learner model types |
| `EXP_GRID_BATCH_SIZE` | List[int] | Query batch sizes |
| `EXP_GRID_RANDOM_SEED` | List[int] | Random seeds for reproducibility |
| `EXP_GRID_START_POINT` | List[int] | Initial labeled sample set indices |
| `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` | List[int] | Train/test split bucket indices |
| `EXP_GRID_NUM_QUERIES` | List[int] | Number of AL iterations |

#### Metrics

| Field | Type | Description |
|-------|------|-------------|
| `METRICS` | List[str] | Metrics to compute per AL cycle |

Available metrics:
- `Standard_ML_Metrics`: Accuracy, F1, precision, recall
- `Selected_Indices`: Which samples were selected
- `Timing_Metrics`: Query selection timing
- `Predicted_Samples`: Model predictions

### Range Syntax

The YAML parser supports range syntax for consecutive integers:

```yaml
# These are equivalent:
EXP_GRID_START_POINT: ['0-4']  # Expands to [0, 1, 2, 3, 4]
EXP_GRID_START_POINT: [0, 1, 2, 3, 4]
```

### Available Strategies

Strategies are prefixed by their source framework:

| Prefix | Framework |
|--------|-----------|
| `ALIPY_*` | ALiPy |
| `LIBACT_*` | libact |
| `SMALLTEXT_*` | small-text |
| `SKACTIVEML_*` | scikit-activeml |
| `PLAYGROUND_*` | playground |
| `OPTIMAL_*` | Oracle strategies |

Examples:
- `ALIPY_RANDOM`, `ALIPY_UNCERTAINTY_LC`, `ALIPY_UNCERTAINTY_ENTROPY`
- `LIBACT_UNCERTAINTY_LC`, `LIBACT_QUIRE`, `LIBACT_DWUS`
- `SMALLTEXT_LEASTCONFIDENCE`, `SMALLTEXT_EMBEDDINGKMEANS`
- `SKACTIVEML_QBC`, `SKACTIVEML_US_MARGIN`
- `OPTIMAL_GREEDY_10`, `OPTIMAL_GREEDY_20`

### Available Learner Models

| Model | Description |
|-------|-------------|
| `RF` | Random Forest |
| `MLP` | Multi-Layer Perceptron |
| `DT` | Decision Tree |
| `RBF_SVM` | SVM with RBF kernel |
| `LINEAR_SVM` | SVM with linear kernel |
| `GNB` | Gaussian Naive Bayes |
| `LR` | Logistic Regression |

---

## Example Configurations

### Minimal Local Test

```yaml
test_minimal:
  EXP_GRID_DATASET: [Iris, wine_origin]
  EXP_GRID_STRATEGY: [ALIPY_RANDOM, ALIPY_UNCERTAINTY_LC]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_BATCH_SIZE: [5]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_START_POINT: [0]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
  EXP_GRID_NUM_QUERIES: [10]
  METRICS: [Standard_ML_Metrics, Selected_Indices, Timing_Metrics]
```

This creates: 2 datasets × 2 strategies × 1 model × 1 batch × 1 seed × 1 start × 1 split × 1 = **4 experiments**

### Paper-Scale Configuration

```yaml
full_experiment:
  EXP_GRID_DATASET:
    - Bioresponse
    - GesturePhaseSegmentationProcessed
    - Iris
    - MiceProtein
    - PenDigits
    # ... 90+ datasets
  EXP_GRID_STRATEGY:
    - ALIPY_RANDOM
    - ALIPY_UNCERTAINTY_LC
    - ALIPY_UNCERTAINTY_MM
    - ALIPY_UNCERTAINTY_ENTROPY
    - ALIPY_GRAPH_DENSITY
    - ALIPY_CORESET_GREEDY
    - ALIPY_DENSITY_WEIGHTED
    - LIBACT_UNCERTAINTY_LC
    - LIBACT_UNCERTAINTY_SM
    - LIBACT_UNCERTAINTY_ENT
    - LIBACT_DWUS
    - LIBACT_QUIRE
    - SMALLTEXT_LEASTCONFIDENCE
    - SMALLTEXT_PREDICTIONENTROPY
    - SMALLTEXT_EMBEDDINGKMEANS
    - SKACTIVEML_QBC
    - SKACTIVEML_US_MARGIN
    - SKACTIVEML_US_LC
    # ... 30+ strategies
  EXP_GRID_LEARNER_MODEL: [MLP, RBF_SVM, RF]
  EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_START_POINT: ['0-99']  # 100 different start points
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: ['0-4']  # 5 different splits
  EXP_GRID_NUM_QUERIES: [100]
  METRICS:
    - Predicted_Samples
    - Selected_Indices
    - Standard_ML_Metrics
    - Timing_Metrics
```

This creates: 90 × 30 × 3 × 6 × 1 × 100 × 5 × 1 = **~24 million experiments**

---

## Config Class Reference

The `Config` class (`misc/config.py`) is the central configuration manager.

### Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUNNING_ENVIRONMENT` | Literal["local", "hpc"] | "local" | Execution environment |
| `EXP_TITLE` | str | Required | Experiment name |
| `WORKER_INDEX` | int | None | Workload row index |
| `RANDOM_SEED` | int | 1312 | Global random seed |
| `N_JOBS` | int | 1 | Parallel workers |
| `EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT` | int | 300 | Per-query timeout |

### Path Attributes (Auto-computed)

| Attribute | Description |
|-----------|-------------|
| `OUTPUT_PATH` | Base output directory |
| `DATASETS_PATH` | Dataset directory |
| `WORKLOAD_FILE_PATH` | Path to workload CSV |
| `CONFIG_FILE_PATH` | Path to saved config |
| `OVERALL_DONE_WORKLOAD_PATH` | Completed experiments |
| `OVERALL_FAILED_WORKLOAD_PATH` | Failed experiments |
| `METRIC_RESULTS_FOLDER` | Per-experiment results |

### Loading Order

1. **CLI Arguments**: Parsed first, highest priority
2. **Server Credentials**: `.server_access_credentials.cfg`
3. **Experiment YAML**: `resources/exp_config.yaml[EXP_TITLE]`
4. **Workload Row**: If `WORKER_INDEX` provided (Step 2 only)

CLI arguments always override other sources.

---

## CLI Arguments

All `Config` attributes can be overridden via CLI:

```bash
# Override experiment title
python 01_create_workload.py --EXP_TITLE my_experiment

# Override environment
python 02_run_experiment.py --EXP_TITLE my_experiment --RUNNING_ENVIRONMENT hpc

# Override paths
python 02_run_experiment.py --EXP_TITLE my_experiment --LOCAL_OUTPUT_PATH /tmp/results
```

### Getting Help

```bash
python 01_create_workload.py --help
```

This prints all available CLI arguments with their types and defaults.
