# Extend the Benchmark

**Add new strategies, datasets, or hyperparameters and integrate them with the shared benchmark.**

---

## Understanding the Configuration

All experiments are defined by the combination of hyperparameters specified in `resources/exp_config.yaml`. The experiment grid uses integer enums from `resources/data_types.py` to compactly represent strategies, datasets, learner models, and other parameters.

### Enums (`resources/data_types.py`)

All entities (strategies, learner models, frameworks, metrics, etc.) are stored as Python `IntEnum` values. This means every strategy, dataset, and model has a unique integer ID used throughout the CSV result files:

!!! warning "Dataset IDs are calculated at runtime"
    Unlike strategies and learner models which have fixed enum IDs in `resources/data_types.py`, **dataset IDs are assigned dynamically at runtime**. The `DATASET` enum is populated by reading `resources/kaggle_datasets.yaml` first, then `resources/openml_datasets.yaml`, and assigning sequential integer IDs based on the order the datasets appear in these YAML files (see [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py)). If you add a new dataset or change the order in a YAML file, the IDs for all subsequent datasets will shift. Keep this in mind when interpreting `EXP_DATASET` values in result files.

```python
from resources.data_types import AL_STRATEGY, LEARNER_MODEL, AL_FRAMEWORK

# Strategies — 76 active learning strategies across 6 frameworks
AL_STRATEGY.ALIPY_RANDOM          # = 1  (paper: "Random")
AL_STRATEGY.ALIPY_UNCERTAINTY_LC  # = 2  (paper: "Uncertainty (LC)")
AL_STRATEGY.ALIPY_CORESET_GREEDY  # = 4  (paper: "CoreSet Greedy")

# Learner models — the ML classifier used inside AL
LEARNER_MODEL.RF         # = 1  (paper: "Random Forest")
LEARNER_MODEL.DT         # = 2  (paper: "Decision Tree")
LEARNER_MODEL.RBF_SVM    # = 5  (paper: "RBF SVM")
LEARNER_MODEL.LINEAR_SVM # = 6  (paper: "Linear SVM")
LEARNER_MODEL.MLP        # = 8  (paper: "MLP")

# Frameworks — which AL library implements the strategy
AL_FRAMEWORK.ALIPY       # = 1
AL_FRAMEWORK.LIBACT      # = 3
AL_FRAMEWORK.SMALLTEXT   # = 5
AL_FRAMEWORK.SKACTIVEML  # = 6
```

When adding a new strategy or model, assign an unused integer ID in the corresponding enum.

### Strategy-to-Class Mapping

Each strategy enum is mapped to its Python implementation class and default hyperparameters in `al_strategy_to_python_classes_mapping` (in `resources/data_types.py`):

```python
# Example: ALIPY_RANDOM maps to QueryInstanceRandom with no extra params
al_strategy_to_python_classes_mapping[AL_STRATEGY.ALIPY_RANDOM] = (
    QueryInstanceRandom, {}
)
```

### Experiment Config (`resources/exp_config.yaml`)

The `exp_config.yaml` file defines named experiment configurations. Each configuration specifies the Cartesian product of hyperparameters to explore:

```yaml
# Example experiment configuration
full_exp_jan:
  EXP_GRID_DATASET: [3, 4, 5, ...]      # Dataset enum IDs
  EXP_GRID_STRATEGY: [1, 2, 4, ...]      # Strategy enum IDs
  EXP_GRID_LEARNER_MODEL: [1, 5, 8]      # Learner model enum IDs (RF, RBF_SVM, MLP)
  EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100]  # Paper: "batch size b"
  EXP_GRID_NUM_QUERIES: [40]             # Paper: "number of AL cycles"
  EXP_GRID_START_POINT: [0-19]          # Start-set index (0–19 for 20 start sets)
  EXP_GRID_RANDOM_SEED: [0]             # For reproducibility
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0-4] # Train/test split index (0–4 for 5 splits)
```

The key hyperparameters and their paper notation:

| Config Key | Paper Notation | Description |
|-----------|----------------|-------------|
| `EXP_GRID_STRATEGY` | Strategy $s$ | Active learning query strategy |
| `EXP_GRID_DATASET` | Dataset $d$ | Dataset used for the experiment |
| `EXP_GRID_LEARNER_MODEL` | Learner model $m$ | ML classifier used inside the AL loop |
| `EXP_GRID_BATCH_SIZE` | Batch size $b$ | Number of samples queried per AL cycle |
| `EXP_GRID_NUM_QUERIES` | Number of queries $q$ | Total number of AL cycles |
| `EXP_GRID_START_POINT` | Start set index | Which pre-generated start set to use (0-indexed) |
| `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` | Train/test split | Which pre-generated train/test split to use (0-indexed) |
| `EXP_GRID_RANDOM_SEED` | Random seed | Seed for reproducibility |

Running `01_create_workload.py` computes the Cartesian product of all these parameters, producing one experiment row per combination in `01_workload.csv`.

### Where config keys come from

OGAL loads configuration from two complementary sources:

- **`resources/exp_config.yaml`** — defines named experiment grids.  Each key in the
  YAML block (e.g. `EXP_GRID_NUM_QUERIES`) maps to an attribute of the `Config` class
  after the section prefix is stripped.  This is the file to edit when adding or changing
  a hyperparameter sweep.
- **`misc/config.py`** — the `Config` class that declares every supported attribute with
  its type and default value.  It also documents the load order (CLI args override the cfg
  file which overrides the YAML which overrides built-in defaults).  Any key you put in
  `exp_config.yaml` must have a matching attribute in `Config` to take effect.

---

## Add a New Batch Size

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100, 200]  # Added 200
```

```bash
python 01_create_workload.py --EXP_TITLE my_experiment
python 02_run_experiment.py --EXP_TITLE my_experiment --WORKER_INDEX 0
```

### Add a New Dataset (OpenML)

```yaml
# resources/openml_datasets.yaml
my_new_dataset:
  data_id: 12345  # OpenML ID
```

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_DATASET: [my_new_dataset, Iris]
```

---

## Add a New AL Strategy

**Step 1:** Add to enum (`resources/data_types.py`):

```python
class AL_STRATEGY(IntEnum):
    MY_CUSTOM_STRATEGY = 100  # Choose unused ID
```

**Step 2:** Add mapping:

```python
al_strategy_to_python_classes_mapping[AL_STRATEGY.MY_CUSTOM_STRATEGY] = (
    MyStrategyClass, {"param": "value"}
)
```

**Step 3:** Use in experiment:

```yaml
EXP_GRID_STRATEGY: [ALIPY_RANDOM, MY_CUSTOM_STRATEGY]
```

---

## Add a New Learner Model

**Step 1:** Add to enum (`resources/data_types.py`):

```python
class LEARNER_MODEL(IntEnum):
    MY_MODEL = 15  # Choose unused ID
```

**Step 2:** Add model initialization logic in the framework runner (e.g., `framework_runners/base_runner.py`).

**Step 3:** Use in experiment:

```yaml
EXP_GRID_LEARNER_MODEL: [1, 5, 8, 15]  # RF, RBF_SVM, MLP, MY_MODEL
```

---

## Validate and Post-Process

```bash
# RESULTS_DIR must match OUTPUT_PATH in the [LOCAL] section of .server_access_credentials.cfg.
# Output paths are controlled entirely by that config file — no environment variable is read by OGAL.
RESULTS_DIR=/absolute/path/to/results  # same value as OUTPUT_PATH in [LOCAL]

# Validate schema
python scripts/validate_results_schema.py --results_path ${RESULTS_DIR}/my_experiment

# Post-process
python 03_calculate_dataset_categorizations.py --EXP_TITLE my_experiment --SAMPLES_CATEGORIZER _ALL --EVA_MODE local
python 04_calculate_advanced_metrics.py --EXP_TITLE my_experiment --COMPUTED_METRICS _ALL --EVA_MODE local

# Run prerequisite scripts
python scripts/convert_y_pred_to_parquet.py --EXP_TITLE my_experiment
python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE my_experiment

# Generate leaderboard (auto-generates _TS/*.parquet if missing)
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

---

## Comparing Results

Run evaluation scripts separately, then compare:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

---

## Next Steps

| Goal | Page |
|------|------|
| Run at HPC scale / Reproduce paper | [Reproduce & Run](reproduce_and_run.md) |
| Development guidelines | [Contributing](../contributing.md) |
