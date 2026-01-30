# Research Reuse Guide

This document explains how to extend OGAL for your own research, including adding new datasets, AL strategies, learner models, and metrics.

## Adding a New Dataset

### Step 1: Add Dataset Definition

Create an entry in `resources/openml_datasets.yaml` or `resources/kaggle_datasets.yaml`:

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

Create `resources/local_datasets.yaml`:

```yaml
my_local_dataset:
  enum_id: 200  # Unique integer ID not used by other datasets
  path: my_local_dataset.csv  # Relative to DATASETS_PATH
```

### Step 2: Prepare Dataset Files

Ensure your dataset is in the correct format:

```
DATASETS_PATH/
├── my_new_dataset.csv          # Features + LABEL_TARGET column
└── my_new_dataset_split.csv    # Train/test split indices
```

**Dataset CSV format** (source: `datasets/__init__.py::load_dataset` - expects `LABEL_TARGET` column):

```csv
feature1,feature2,...,featureN,LABEL_TARGET
0.5,1.2,...,0.8,0
0.3,0.9,...,0.4,1
...
```

**Split CSV format:**

```csv
train,test,start_points
"[0, 1, 2, ...]","[100, 101, ...]","[[0, 5, 10], [1, 6, 11], ...]"
```

### Step 3: Update Enum (Optional)

If using local datasets, they're automatically added to the `DATASET` enum via `extend_enum()` in `datasets/__init__.py`.

### Step 4: Use in Experiments

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_DATASET: [my_new_dataset, Iris, wine_origin]
  ...
```

---

## Adding a New AL Strategy

### Step 1: Choose or Create a Framework Runner

OGAL supports multiple AL frameworks (source: `framework_runners/` directory):

| Framework | Runner File |
|-----------|-------------|
| ALiPy | `framework_runners/alipy_runner.py` |
| libact | `framework_runners/libact_runner.py` |
| small-text | `framework_runners/smalltext_runner.py` |
| scikit-activeml | `framework_runners/skactiveml_runner.py` |
| playground | `framework_runners/playground_runner.py` |
| optimal (oracle) | `framework_runners/optimal_runner.py` |

### Step 2: Add Strategy to Enum

In `resources/data_types.py`:

```python
@unique
class AL_STRATEGY(IntEnum):
    # ... existing strategies ...
    MY_NEW_STRATEGY = 100  # Choose unused ID
```

### Step 3: Add Strategy Mapping

In `resources/data_types.py`:

```python
al_strategy_to_python_classes_mapping: Dict[AL_STRATEGY, Tuple[Callable, Dict[Any, Any]]] = {
    # ... existing mappings ...
    AL_STRATEGY.MY_NEW_STRATEGY: (
        MyStrategyClass,  # The query strategy class
        {"param1": "value1"}  # Default parameters
    ),
}
```

### Step 4: Implement in Runner (if needed)

If using a new framework, create a new runner in `framework_runners/`:

```python
# framework_runners/my_framework_runner.py
from framework_runners.base_runner import AL_Experiment

class MY_FRAMEWORK_AL_Experiment(AL_Experiment):
    """
    Custom AL framework adapter.
    Base class: framework_runners/base_runner.py::AL_Experiment
    """
    def get_AL_strategy(self):
        # Initialize your strategy
        self.strategy = MyStrategyClass(**params)
    
    def query_AL_strategy(self) -> SampleIndiceList:
        # Return list of selected sample indices
        return self.strategy.query(
            X=self.local_X_train,
            y=self.local_Y_train,
            labeled=self.local_train_labeled_idx,
            batch_size=self.config.EXP_BATCH_SIZE
        )
    
    def prepare_dataset(self):
        # Framework-specific dataset preparation
        pass
```

### Step 5: Register in 02_run_experiment.py

```python
# 02_run_experiment.py
from framework_runners.my_framework_runner import MY_FRAMEWORK_AL_Experiment

if str(config.EXP_STRATEGY.name).startswith("MY_FRAMEWORK"):
    al_experiment = MY_FRAMEWORK_AL_Experiment(config)
```

---

## Adding a New Learner Model

### Step 1: Add to Enum

In `resources/data_types.py`:

```python
@unique
class LEARNER_MODEL(IntEnum):
    # ... existing models ...
    MY_MODEL = 50  # Choose unused ID
```

### Step 2: Add Mapping

```python
learner_models_to_classes_mapping: Dict[LEARNER_MODEL, Tuple[Callable, Dict[Any, Any]]] = {
    # ... existing mappings ...
    LEARNER_MODEL.MY_MODEL: (
        MyModelClass,  # sklearn-compatible classifier
        {
            "param1": "value1",
            "param2": "value2"
        }
    ),
}
```

### Requirements

The model class must be scikit-learn compatible:

- `fit(X, y)` method
- `predict(X)` method
- `predict_proba(X)` method (for uncertainty-based strategies)

---

## Adding a New Metric

### Step 1: Create Metric Class

In `metrics/`:

```python
# metrics/My_New_Metric.py
from metrics.base_metric import Base_Metric

class My_New_Metric(Base_Metric):
    metrics = ["my_metric_name"]  # List of metric names to track
    
    def post_retraining_of_learner_hook(self, al_experiment):
        # Compute your metric after each model retraining
        value = self._compute_metric(al_experiment)
        self.metric_values["my_metric_name"].append(value)
    
    def _compute_metric(self, al_experiment):
        # Your metric computation logic
        return some_value
```

### Step 2: Register Metric

Add to experiment YAML:

```yaml
my_experiment:
  METRICS: [Standard_ML_Metrics, My_New_Metric]
```

### Available Hooks

| Hook | When Called |
|------|-------------|
| `pre_query_selection_hook` | Before AL strategy selects samples |
| `post_query_selection_hook` | After selection, before training |
| `pre_retraining_of_learner_hook` | Before model.fit() |
| `post_retraining_of_learner_hook` | After model.fit() |
| `save_metrics` | At experiment end |

---

## What to Keep Fixed vs Vary

Based on OGAL's experimental design:

### Keep Fixed for Fair Comparisons

| Parameter | Recommendation |
|-----------|----------------|
| `EXP_NUM_QUERIES` | Same number of AL iterations |
| Train/test split | Use same `EXP_TRAIN_TEST_BUCKET_SIZE` range |
| Random seeds | Use same `EXP_RANDOM_SEED` |
| Start points | Use same `EXP_START_POINT` range |

### Vary for Sensitivity Analysis

| Parameter | Purpose |
|-----------|---------|
| `EXP_BATCH_SIZE` | Study batch size sensitivity |
| `EXP_LEARNER_MODEL` | Study learner impact |
| `EXP_START_POINT` | Study initialization sensitivity |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | Study data split sensitivity |

---

## Reproducibility Checklist

### Environment

- [ ] Use exact conda lock file (`conda-linux-64.lock`)
- [ ] Pin Poetry dependencies (`poetry.lock`)
- [ ] Record Python version (3.11)

### Configuration

- [ ] Save `00_config.yaml` with git commit hash
- [ ] Document any local modifications to config
- [ ] Archive the exact `resources/exp_config.yaml` used

### Seeds

- [ ] Use fixed random seeds (`EXP_GRID_RANDOM_SEED`)
- [ ] Document NumPy/Python random state handling
- [ ] Note any non-deterministic operations

### Data

- [ ] Record dataset versions/sources
- [ ] Archive train/test splits (`*_split.csv`)
- [ ] Document any preprocessing steps

### Code

- [ ] Tag git commit for each experiment run
- [ ] Save modified files if not committed
- [ ] Document framework versions (ALiPy, libact, etc.)

---

## Using the OPARA Archived Results

The complete experiment results (4.6M+ hyperparameter combinations) are archived at **[DOI: 10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)**.

### What's in the Archive

The OPARA archive (`full_exp_jan.zip`) contains:

| Content | Description |
|---------|-------------|
| **Raw experiment results** | Per-cycle metrics for all 4.6M hyperparameter combinations |
| **28 AL strategies** | From ALiPy, libact, scikit-activeml, small-text, playground |
| **92 datasets** | Binary and multi-class classification tasks |
| **Computed metrics** | AUC, ramp-up, plateau, time-lag, distance metrics |
| **Dataset categorizations** | Per-sample hardness metrics |
| **Workload files** | Complete experiment definitions |

### Archive Structure

```
full_exp_jan/
├── ALIPY_RANDOM/              # Strategy results
│   ├── Iris/                  # Dataset results
│   │   ├── accuracy.csv.xz
│   │   ├── weighted_f1-score.csv.xz
│   │   ├── full_auc_accuracy.csv.xz
│   │   └── ...
│   ├── wine_origin/
│   └── ...
├── ALIPY_UNCERTAINTY_LC/
├── SKACTIVEML_RANDOM/
├── ... (28 strategies total)
├── 05_done_workload.csv       # 4.5M+ completed experiments
└── 05_failed_workloads.csv    # Failed experiments with errors
```

### Integration with OGAL

The archive format matches OGAL's output exactly. To use the archived data:

#### 1. Download and Extract

```bash
# Download from OPARA portal (requires manual download)
# Extract selectively (full archive is several TB):
unzip -q full_exp_jan.zip "full_exp_jan/ALIPY_RANDOM/*" "full_exp_jan/05_*.csv"
```

#### 2. Configure OGAL to Point to Archived Data

```ini
# .server_access_credentials.cfg
[LOCAL]
LOCAL_OUTPUT_PATH = /path/to/full_exp_jan/
```

#### 3. Run Evaluation Scripts on Archived Data

```bash
# Regenerate paper leaderboard rankings
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan

# Compute metric correlations
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan

# Analyze runtime
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

### Building on the Archived Results

#### Compare New Strategies Against Archive

```python
import pandas as pd

# Load archived results
archived_accuracy = pd.read_csv(
    "full_exp_jan/ALIPY_RANDOM/Iris/full_auc_accuracy.csv.xz"
)

# Load your new experiment results
your_results = pd.read_csv(
    "your_exp/YOUR_NEW_STRATEGY/Iris/full_auc_accuracy.csv.xz"
)

# Merge on EXP_UNIQUE_ID (matching hyperparameter combinations)
comparison = archived_accuracy.merge(
    your_results, 
    on="EXP_UNIQUE_ID", 
    suffixes=("_archived", "_new")
)
```

#### Extend with New Datasets

Run OGAL experiments on new datasets using the same hyperparameter grid, then combine with archived results for extended analysis.

### File Size Considerations

| Subset | Approximate Size |
|--------|-----------------|
| Single strategy/dataset | ~50-500 MB |
| All results for one strategy | ~5-50 GB |
| Complete archive | Several TB |
| Dense workload subset | ~100 GB |

**Tip:** Start with the dense workload subset (`06_dense_workload.csv`) which contains ~1.1M complete hyperparameter combinations with no missing values.

---

## Code Pointers

### Key Files for Extension

| Purpose | File |
|---------|------|
| Dataset loading | `datasets/__init__.py` (DATASET enum, loader classes) |
| AL strategy dispatch | `02_run_experiment.py` (main experiment runner) |
| Strategy definitions | `resources/data_types.py` (AL_STRATEGY enum, mappings) |
| Runner base class | `framework_runners/base_runner.py` (AL_Experiment abstract class) |
| Metric base class | `metrics/base_metric.py` (Base_Metric abstract class) |
| Configuration | `misc/config.py` (Config class) |

### Understanding the AL Loop

The active learning loop is implemented in `framework_runners/base_runner.py::AL_Experiment`:

```
02_run_experiment.py
    └── AL_Experiment.run_experiment()
        └── for iteration in range(total_iterations):
            └── al_cycle()
                ├── metric.pre_query_selection_hook()
                ├── query_AL_strategy()  # Select samples
                ├── metric.post_query_selection_hook()
                ├── metric.pre_retraining_of_learner_hook()
                ├── model.fit()  # Retrain
                └── metric.post_retraining_of_learner_hook()
```

(source: `framework_runners/base_runner.py::AL_Experiment.al_cycle`)

---

## Example: Complete New Strategy

Here's a complete example of adding a custom uncertainty sampling variant:

### 1. Add to Enum

```python
# resources/data_types.py
class AL_STRATEGY(IntEnum):
    # ...
    CUSTOM_UNCERTAINTY_WEIGHTED = 101
```

### 2. Implement Strategy

```python
# optimal_query_strategies/custom_uncertainty.py
import numpy as np

class CustomUncertaintyWeighted:
    def __init__(self, weight_factor=0.5):
        self.weight_factor = weight_factor
    
    def query(self, X, y_pred_proba, unlabeled_idx, batch_size):
        # Compute uncertainty scores
        uncertainties = 1 - np.max(y_pred_proba[unlabeled_idx], axis=1)
        
        # Weight by sample density (example)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X[unlabeled_idx])
        distances, _ = nn.kneighbors()
        densities = 1 / (distances.mean(axis=1) + 1e-6)
        
        # Combined score
        scores = uncertainties * self.weight_factor + densities * (1 - self.weight_factor)
        
        # Select top-k
        top_k_local = np.argsort(scores)[-batch_size:]
        return [unlabeled_idx[i] for i in top_k_local]
```

### 3. Add Mapping

```python
# resources/data_types.py
from optimal_query_strategies.custom_uncertainty import CustomUncertaintyWeighted

al_strategy_to_python_classes_mapping[AL_STRATEGY.CUSTOM_UNCERTAINTY_WEIGHTED] = (
    CustomUncertaintyWeighted,
    {"weight_factor": 0.5}
)
```

### 4. Create Runner Method

```python
# framework_runners/optimal_runner.py
def query_AL_strategy(self) -> SampleIndiceList:
    if isinstance(self.strategy, CustomUncertaintyWeighted):
        y_pred_proba = self.model.predict_proba(self.local_X_train)
        return self.strategy.query(
            X=self.local_X_train,
            y_pred_proba=y_pred_proba,
            unlabeled_idx=self.local_train_unlabeled_idx,
            batch_size=self.config.EXP_BATCH_SIZE
        )
```

### 5. Use in Experiment

```yaml
my_custom_experiment:
  EXP_GRID_STRATEGY: [CUSTOM_UNCERTAINTY_WEIGHTED, ALIPY_RANDOM]
  # ...
```
