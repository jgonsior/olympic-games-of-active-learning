# Extend the Benchmark

**You're a researcher who wants to add new AL strategies, datasets, learner models, or hyperparameters to the benchmark—and contribute back to the shared dataset.**

This guide walks you through every type of extension with concrete examples.

---

## What You Can Add

| Extension Type | Difficulty | Time |
|----------------|------------|------|
| New hyperparameter values | Easy | Minutes |
| New dataset | Easy | ~30 min |
| New learner model | Medium | ~1 hour |
| New AL strategy (existing framework) | Medium | ~1-2 hours |
| New AL framework | Hard | ~1 day |

---

## Prerequisites

- [Understand the Codebase](understand_codebase.md) — Know where things live
- Development environment set up (see [Contributing](../contributing.md))

---

## Adding New Hyperparameter Values

The easiest extension: modify the experiment grid.

### Example: Add a New Batch Size

In `resources/exp_config.yaml`:

```yaml
my_experiment:
  EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100, 200]  # Added 200
```

Then run:

```bash
python 01_create_workload.py --EXP_TITLE my_experiment
python 02_run_experiment.py --EXP_TITLE my_experiment --WORKER_INDEX 0
```

---

## Adding a New Dataset

### Step 1: Add Dataset Definition

**For OpenML datasets:**

```yaml
# resources/openml_datasets.yaml
my_new_dataset:
  data_id: 12345  # OpenML dataset ID
```

**For Kaggle datasets:**

```yaml
# resources/kaggle_datasets.yaml
my_kaggle_dataset:
  kaggle_name: username/dataset-name
```

**For local datasets:**

```yaml
# resources/local_datasets.yaml (create if doesn't exist)
my_local_dataset:
  enum_id: 200  # Unique ID not used by other datasets
  path: my_local_dataset.csv  # Relative to DATASETS_PATH
```

### Step 2: Prepare Dataset Files

Place your dataset in `DATASETS_PATH/`:

```
DATASETS_PATH/
├── my_local_dataset.csv          # Features + LABEL_TARGET column
└── my_local_dataset_split.csv    # Train/test split indices
```

**Dataset CSV format:**

```csv
feature1,feature2,...,featureN,LABEL_TARGET
0.5,1.2,...,0.8,0
0.3,0.9,...,0.4,1
```

### Step 3: Use in Experiments

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_DATASET: [my_new_dataset, Iris]
```

---

## Adding a New AL Strategy

### Step 1: Add to Enum

In `resources/data_types.py`:

```python
@unique
class AL_STRATEGY(IntEnum):
    # ... existing strategies ...
    MY_CUSTOM_STRATEGY = 100  # Choose unused ID
```

!!! tip "Finding an unused ID"
    ```python
    from resources.data_types import AL_STRATEGY
    print(max(s.value for s in AL_STRATEGY) + 1)
    ```

### Step 2: Implement the Strategy

If using an existing framework (ALiPy, libact, etc.), you can often reference existing classes.

For a custom strategy, create a new file:

```python
# optimal_query_strategies/my_custom_strategy.py
import numpy as np

class MyCustomStrategy:
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def query(self, X_unlabeled, model, batch_size):
        probs = model.predict_proba(X_unlabeled)
        uncertainty = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        return np.argsort(uncertainty)[-batch_size:]
```

### Step 3: Add Strategy Mapping

In `resources/data_types.py`:

```python
from optimal_query_strategies.my_custom_strategy import MyCustomStrategy

al_strategy_to_python_classes_mapping[AL_STRATEGY.MY_CUSTOM_STRATEGY] = (
    MyCustomStrategy,
    {"temperature": 0.5}  # Default parameters
)
```

### Step 4: Register in Runner (if needed)

If using a custom framework, update the runner to handle your strategy:

```python
# In framework_runners/optimal_runner.py
def get_AL_strategy(self):
    if self.config.EXP_STRATEGY == AL_STRATEGY.MY_CUSTOM_STRATEGY:
        strategy_class, params = al_strategy_to_python_classes_mapping[
            AL_STRATEGY.MY_CUSTOM_STRATEGY
        ]
        self.strategy = strategy_class(**params)
```

### Step 5: Use in Experiment

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_STRATEGY: [ALIPY_RANDOM, MY_CUSTOM_STRATEGY]
```

---

## Adding a New Learner Model

### Step 1: Add to Enum

In `resources/data_types.py`:

```python
@unique
class LEARNER_MODEL(IntEnum):
    RF = 1
    MLP = 2
    SVM = 3
    MY_NEW_MODEL = 10  # Choose unused ID
```

### Step 2: Add Mapping

```python
from sklearn.ensemble import GradientBoostingClassifier

learner_model_to_python_classes_mapping = {
    # ... existing mappings ...
    LEARNER_MODEL.MY_NEW_MODEL: (GradientBoostingClassifier, {"n_estimators": 100}),
}
```

### Requirements

The model class must be scikit-learn compatible:

- `fit(X, y)` method
- `predict(X)` method  
- `predict_proba(X)` method (for uncertainty-based strategies)

---

## Running Your Extended Experiments

### Local Execution

```bash
# Create workload
python 01_create_workload.py --EXP_TITLE my_experiment

# Run one experiment
python 02_run_experiment.py --EXP_TITLE my_experiment --WORKER_INDEX 0

# Verify output
ls ${OGAL_OUTPUT}/my_experiment/
```

### HPC Execution

```bash
# Create workload (generates SLURM script)
python 01_create_workload.py --EXP_TITLE my_experiment

# Submit to SLURM
sbatch ${OGAL_OUTPUT}/my_experiment/02_slurm.slurm
```

---

## Validating Your Results

### Step 1: Check Schema Compliance

```bash
python scripts/validate_results_schema.py --results_path ${OGAL_OUTPUT}/my_experiment
```

### Step 2: Check for Duplicates

```bash
python scripts/validate_results_schema.py \
    --results_path ${OGAL_OUTPUT}/my_experiment \
    --compare_with ${OGAL_OUTPUT}/full_exp_jan
```

### Step 3: Run Post-Processing

```bash
# Dataset categorizations
python 03_calculate_dataset_categorizations.py \
    --EXP_TITLE my_experiment \
    --SAMPLES_CATEGORIZER _ALL \
    --EVA_MODE local

# Advanced metrics
python 04_calculate_advanced_metrics.py \
    --EXP_TITLE my_experiment \
    --COMPUTED_METRICS _ALL \
    --EVA_MODE local
```

### Step 4: Generate Leaderboard

```bash
python -m eva_scripts.learning_curve --EXP_TITLE my_experiment
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

---

## Merging with Existing Results

### Option A: Keep Separate (Recommended)

Run evaluation scripts separately and compare:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

### Option B: Merge Workloads

!!! warning "Back up first"
    Always back up existing data before merging.

```bash
python -m scripts.merge_two_workloads \
    --EXP_TITLE full_exp_jan \
    --SECOND_MERGE_PATH ${OGAL_OUTPUT}/my_experiment
```

---

## Contributing Back

To contribute your results to the shared benchmark:

1. **Run validation** — Ensure schema compliance
2. **Document your extensions** — Update `00_config.yaml` with provenance info
3. **Open a GitHub issue** — Describe what you added
4. **Share results** — Upload to a data repository

See [Data Enrichment Protocol](../reference/data_enrichment.md) for full details.

---

## Next Steps

| Goal | Page |
|------|------|
| Run experiments at scale | [Runbook](../reference/runbook.md) |
| Understand the data schema | [Results Schema](../reference/results_schema.md) |
| Add more complex extensions | [Research Reuse Guide](../reference/research_reuse.md) |
| Development guidelines | [Contributing](../contributing.md) |
