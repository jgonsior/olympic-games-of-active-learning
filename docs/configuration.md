# Configuration Guide

OGAL uses a shared configuration system that controls all aspects of the experimental pipeline. This document explains the configuration files, their structure, and how they are loaded by each script.

## Paper Terminology Reference

The OGAL configuration parameters map to the notation used in the research paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)):

| Paper Symbol | Paper Term | OGAL Config Parameter | Description |
|-------------|------------|----------------------|-------------|
| 𝔻 | Dataset | `EXP_GRID_DATASET` | Dataset identifiers |
| 𝕊 | AL Strategy | `EXP_GRID_STRATEGY` | Query strategy selection |
| 𝕃 | Learner Model | `EXP_GRID_LEARNER_MODEL` | ML classification model |
| 𝔹 | Batch Size | `EXP_GRID_BATCH_SIZE` | Samples queried per AL cycle |
| 𝕋 | Train-Test-Split | `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` | Data partitioning |
| 𝕀 | Initial Start Set | `EXP_GRID_START_POINT` | Initial labeled samples |
| 𝕄 | Metric | `METRICS` | Evaluation metrics |
| c | AL Cycles | `EXP_GRID_NUM_QUERIES` | Number of iterations |

The paper defines a single AL experiment as:

> **E = (𝒮, D, 𝒯, ℐ, M, b, c, ℒ)**
>
> A combination of hyperparameters for simulating one AL strategy on a dataset.

The experimental grid is the Cartesian product: **𝕊 × 𝔻 × 𝕋 × 𝕀 × 𝔹 × 𝕃**

---

## Configuration Files Overview

| File | Purpose | Format |
|------|---------|--------|
| `.server_access_credentials.cfg` | Machine-specific paths (local vs HPC) | INI |
| [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml) | Experiment grid definitions | YAML |
| `OUTPUT_PATH/<EXP_TITLE>/00_config.yaml` | Saved experiment configuration | YAML |

## Server Access Credentials

**File:** `.server_access_credentials.cfg` (gitignored)

This file defines paths for both local development and HPC cluster environments. The `RUNNING_ENVIRONMENT` setting determines which section is active (default: `local`; source: [`misc/config.py::Config.RUNNING_ENVIRONMENT`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)).

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

The `Config` class in [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) loads this file via `_load_server_setup_from_file()` (source: [`misc/config.py::Config._load_server_setup_from_file`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)):

1. Parses both `[HPC]` and `[LOCAL]` sections
2. Prefixes each key with section name (e.g., `HPC_DATASETS_PATH`, `LOCAL_DATASETS_PATH`)
3. Sets attributes on the Config instance
4. The `_pathes_magic()` method then selects the appropriate paths based on `RUNNING_ENVIRONMENT` (source: [`misc/config.py::Config._pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

---

## Experiment Configuration

**File:** [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml)

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

These define the Cartesian product for the experiment workload. Each maps to a hyperparameter from the paper's notation:

| Field | Type | Paper Symbol | Description |
|-------|------|--------------|-------------|
| `EXP_GRID_DATASET` | List | 𝔻 | Dataset identifiers (names or IDs) |
| `EXP_GRID_STRATEGY` | List | 𝕊 | AL query strategy names |
| `EXP_GRID_LEARNER_MODEL` | List | 𝕃 | Learner model types |
| `EXP_GRID_BATCH_SIZE` | List[int] | 𝔹 | Query batch sizes (samples per AL cycle) |
| `EXP_GRID_RANDOM_SEED` | List[int] | - | Random seeds for reproducibility |
| `EXP_GRID_START_POINT` | List[int] | 𝕀 | Initial labeled set indices |
| `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` | List[int] | 𝕋 | Train/test split bucket indices |
| `EXP_GRID_NUM_QUERIES` | List[int] | c | Number of AL cycles/iterations |

#### Metrics (𝕄 in paper)

| Field | Type | Description |
|-------|------|-------------|
| `METRICS` | List[str] | Metrics to compute per AL cycle |

Available metrics (corresponds to paper's aggregation-metrics):

- `Standard_ML_Metrics`: Accuracy, F1-score, precision, recall (source: [`metrics/Standard_ML_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Standard_ML_Metrics.py) - standard ML metrics from paper)
- `Selected_Indices`: Which samples were queried (R(Q) in paper notation; source: [`metrics/Selected_Indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Selected_Indices.py))
- `Timing_Metrics`: Query selection timing (source: [`metrics/Timing_Metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Timing_Metrics.py))
- `Predicted_Samples`: Model predictions (source: [`metrics/Predicted_Samples.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Predicted_Samples.py))

The paper discusses several aggregation-metrics for evaluating learning curves:
- **Full AUC**: Area under entire learning curve
- **Ramp-up AUC**: Early-phase performance (initial AL cycles)
- **Plateau AUC**: Late-phase performance (saturation)
- **Final Value**: Last AL cycle's metric value
- **First-5 / Last-5**: Mean of first/last 5 iterations

These are computed by [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py).

### Range Syntax

The YAML parser supports range syntax for consecutive integers (source: [`misc/config.py::Config._load_from_yaml_file`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), lines ~350-355):

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

The `Config` class ([`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)) is the central configuration manager (source: [`misc/config.py::Config`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)).

### Key Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `RUNNING_ENVIRONMENT` | Literal["local", "hpc"] | "local" | Execution environment (source: [`misc/config.py::Config.RUNNING_ENVIRONMENT`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)) |
| `EXP_TITLE` | str | Required | Experiment name |
| `WORKER_INDEX` | int | None | Workload row index |
| `RANDOM_SEED` | int | 1312 | Global random seed (source: [`misc/config.py::Config.RANDOM_SEED`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)) |
| `N_JOBS` | int | 1 | Parallel workers |
| `EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT` | int | 300 | Per-query timeout (source: [`misc/config.py::Config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)) |

### Path Attributes (Auto-computed)

| Attribute | Description | Source |
|-----------|-------------|--------|
| `OUTPUT_PATH` | Base output directory | [`misc/config.py::Config._pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `DATASETS_PATH` | Dataset directory | [`misc/config.py::Config._pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `WORKLOAD_FILE_PATH` | Path to workload CSV | [`misc/config.py::Config.WORKLOAD_FILE_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `CONFIG_FILE_PATH` | Path to saved config | [`misc/config.py::Config.CONFIG_FILE_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `OVERALL_DONE_WORKLOAD_PATH` | Completed experiments | [`misc/config.py::Config.OVERALL_DONE_WORKLOAD_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `OVERALL_FAILED_WORKLOAD_PATH` | Failed experiments | [`misc/config.py::Config.OVERALL_FAILED_WORKLOAD_PATH`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| `METRIC_RESULTS_FOLDER` | Per-experiment results | [`misc/config.py::Config.METRIC_RESULTS_FOLDER`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |

### Loading Order

Configuration is loaded in this priority order (source: [`misc/config.py::Config.__init__`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py)):

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
