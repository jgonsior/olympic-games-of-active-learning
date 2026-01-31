# Research Reuse Guide

This document explains how to extend OGAL for your own research, including adding new datasets, AL strategies, learner models, and metrics.

## Adding a New Dataset

### Step 1: Add Dataset Definition

Create an entry in [`resources/openml_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/openml_datasets.yaml) or [`resources/kaggle_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/kaggle_datasets.yaml):

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

**Dataset CSV format** (source: [`datasets/__init__.py::load_dataset`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py) - expects `LABEL_TARGET` column):

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

If using local datasets, they're automatically added to the `DATASET` enum via `extend_enum()` in [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py).

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
| ALiPy | [`framework_runners/alipy_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/alipy_runner.py) |
| libact | [`framework_runners/libact_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/libact_runner.py) |
| small-text | [`framework_runners/smalltext_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/smalltext_runner.py) |
| scikit-activeml | [`framework_runners/skactiveml_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/skactiveml_runner.py) |
| playground | [`framework_runners/playground_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/playground_runner.py) |
| optimal (oracle) | [`framework_runners/optimal_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/optimal_runner.py) |

### Step 2: Add Strategy to Enum

In [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py):

```python
@unique
class AL_STRATEGY(IntEnum):
    # ... existing strategies ...
    MY_NEW_STRATEGY = 100  # Choose unused ID
```

### Step 3: Add Strategy Mapping

In [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py):

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

If using a new framework, create a new runner in [`framework_runners/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners):

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

```yaml
# resources/kaggle_datasets.yaml
my_kaggle_dataset:
  kaggle_name: username/dataset-name
```

---

## Adding a New Learner Model

### Step 1: Add to Enum

In [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py):

```python
@unique
class LEARNER_MODEL(IntEnum):
    RF = 1    # Random Forest
    MLP = 2   # Multi-Layer Perceptron
    SVM = 3   # Support Vector Machine
    MY_NEW_MODEL = 10  # Your new model
```

### Step 2: Add Mapping

```python
learner_model_to_python_classes_mapping = {
    LEARNER_MODEL.RF: (RandomForestClassifier, {"n_estimators": 100}),
    LEARNER_MODEL.MLP: (MLPClassifier, {"max_iter": 500}),
    LEARNER_MODEL.SVM: (SVC, {"probability": True}),
    LEARNER_MODEL.MY_NEW_MODEL: (MyCustomClassifier, {"param": "value"}),
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

In [`metrics/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics):

```python
# metrics/My_Custom_Metric.py
from metrics.base_metric import Base_Metric

class My_Custom_Metric(Base_Metric):
    def __init__(self):
        super().__init__()
        self.metric_name = "my_custom_metric"
    
    def post_retraining_of_learner_hook(self, experiment):
        """Compute metric after each AL iteration."""
        # Your computation here
        value = compute_my_metric(experiment.y_test, experiment.y_pred)
        self.results.append(value)
```

### Step 2: Register Metric

Add to the metrics list in your experiment configuration:

```python
# In 02_run_experiment.py or experiment setup
from metrics.My_Custom_Metric import My_Custom_Metric

experiment.metrics.append(My_Custom_Metric())
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
- [ ] Archive the exact [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml) used

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

??? note "Convenience link (may change)"
    Direct download (not guaranteed stable):
    ```
    https://opara.zih.tu-dresden.de/xmlui/bitstream/handle/123456789/5678/full_exp_jan.zip
    ```

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
├── <STRATEGY>/<DATASET>/
│   ├── accuracy.csv.xz
│   ├── weighted_f1-score.csv.xz
│   ├── full_auc_*.csv.xz
│   └── ...
├── 05_done_workload.csv
└── 01_workload.csv
```

### Integration with OGAL

The archive format matches OGAL's output exactly. To use the archived data:

#### 1. Download and Extract

```bash
# Download from DOI landing page
wget <URL_FROM_DOI_LANDING_PAGE>

# Extract
export OGAL_OUTPUT=/path/to/results
unzip full_exp_jan.zip -d ${OGAL_OUTPUT}/
```

#### 2. Configure OGAL to Point to Archived Data

```bash
cat > .server_access_credentials.cfg << EOF
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF
```

#### 3. Run Evaluation Scripts on Archived Data

```bash
# Generate leaderboard
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

!!! tip "Sanity check"
    If `plots/final_leaderboard/` is empty, verify `OUTPUT_PATH` in your config points to the extracted archive.

### Building on the Archived Results

#### Compare New Strategies Against Archive

```bash
# Run your new experiments
python 01_create_workload.py --EXP_TITLE my_new_strategy
python 02_run_experiment.py --EXP_TITLE my_new_strategy --WORKER_INDEX 0

# Generate leaderboards for both
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE my_new_strategy
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
| Dataset loading | [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py) (DATASET enum, loader classes) |
| AL strategy dispatch | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) (main experiment runner) |
| Strategy definitions | [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) (AL_STRATEGY enum, mappings) |
| Runner base class | [`framework_runners/base_runner.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py) (AL_Experiment abstract class) |
| Metric base class | [`metrics/base_metric.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/base_metric.py) (Base_Metric abstract class) |
| Configuration | [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) (Config class) |

### Understanding the AL Loop

The active learning loop is implemented in [`framework_runners/base_runner.py::AL_Experiment`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py):

```python
# Simplified AL loop structure
for iteration in range(num_queries):
    # 1. Query strategy selects samples
    selected_indices = self.query_AL_strategy()
    
    # 2. Update labeled set
    self.local_train_labeled_idx.extend(selected_indices)
    
    # 3. Retrain model
    self.learner.fit(X_labeled, y_labeled)
    
    # 4. Record metrics
    for metric in self.metrics:
        metric.post_retraining_of_learner_hook(self)
```

(source: [`framework_runners/base_runner.py::AL_Experiment.al_cycle`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py))

---

## Example: Complete New Strategy

Here's a complete example of adding a custom uncertainty sampling variant:

### 1. Add to Enum

```python
# In resources/data_types.py
@unique
class AL_STRATEGY(IntEnum):
    # ... existing strategies ...
    MY_CUSTOM_UNCERTAINTY = 100
```

### 2. Implement Strategy

```python
# In optimal_query_strategies/my_custom_uncertainty.py
import numpy as np

class MyCustomUncertainty:
    """Custom uncertainty sampling with temperature scaling."""
    
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def query(self, X_unlabeled, model, batch_size):
        # Get prediction probabilities
        probs = model.predict_proba(X_unlabeled)
        
        # Apply temperature scaling
        scaled_probs = probs ** (1 / self.temperature)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        
        # Compute uncertainty (entropy)
        uncertainty = -np.sum(scaled_probs * np.log(scaled_probs + 1e-10), axis=1)
        
        # Return indices of most uncertain samples
        return np.argsort(uncertainty)[-batch_size:]
```

### 3. Add Mapping

```python
# In resources/data_types.py
from optimal_query_strategies.my_custom_uncertainty import MyCustomUncertainty

al_strategy_to_python_classes_mapping[AL_STRATEGY.MY_CUSTOM_UNCERTAINTY] = (
    MyCustomUncertainty,
    {"temperature": 0.5}  # Default parameters
)
```

### 4. Create Runner Method

```python
# In framework_runners/optimal_runner.py (or create new runner)
def get_AL_strategy(self):
    if self.config.EXP_STRATEGY == AL_STRATEGY.MY_CUSTOM_UNCERTAINTY:
        strategy_class, params = al_strategy_to_python_classes_mapping[
            AL_STRATEGY.MY_CUSTOM_UNCERTAINTY
        ]
        self.strategy = strategy_class(**params)
```

### 5. Use in Experiment

```yaml
# In resources/exp_config.yaml
my_experiment:
  EXP_GRID_STRATEGY:
    - ALIPY_RANDOM
    - MY_CUSTOM_UNCERTAINTY
  EXP_GRID_DATASET: [Iris, wine_origin]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_BATCH_SIZE: [5, 10]
```
