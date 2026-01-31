# Dataset Metadata & Auto-Computed Categorizations

This document explains the automatically computed dataset categorizations in OGAL — per-sample characteristics that describe the "hardness" or properties of individual samples. These are used by evaluation scripts to analyze strategy behavior.

---

## Overview

OGAL computes per-sample characteristics for each dataset that help explain:
- Why certain samples are harder to classify
- Which samples are most informative for AL strategies
- How sample properties relate to strategy performance

These categorizations are computed **automatically** by `03_calculate_dataset_categorizations.py` and stored as compressed NumPy arrays.

---

## Why Auto-Computed Categorizations?

| Reason | Explanation |
|--------|-------------|
| **Interpretability** | Understand why strategies select certain samples |
| **Strategy comparison** | Compare what types of samples each strategy queries |
| **Dataset characterization** | Profile datasets by their sample difficulty distribution |
| **Analysis automation** | Enable systematic correlation studies |

---

## Computation Process

### Script

**Source:** `03_calculate_dataset_categorizations.py`

### Invocation

```bash
# Create workloads for all categorizers on all datasets
python 03_calculate_dataset_categorizations.py --EXP_TITLE your_experiment \
    --SAMPLES_CATEGORIZER _ALL --EVA_MODE create

# Run locally
python 03_calculate_dataset_categorizations.py --EXP_TITLE your_experiment \
    --SAMPLES_CATEGORIZER _ALL --EVA_MODE local

# Run on SLURM
python 03_calculate_dataset_categorizations.py --EXP_TITLE your_experiment \
    --SAMPLES_CATEGORIZER _ALL --EVA_MODE slurm
```

(source: `03_calculate_dataset_categorizations.py`, lines 27-47)

### Core Function

**Source:** `metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`

```python
def categorize_samples(self, dataset: DATASET) -> None:
    samples_categorization_path = Path(
        f"{self.config.OUTPUT_PATH}/_{self.__class__.__name__}/{dataset.name}.npz"
    )
    
    if not self.config.OVERWRITE_EXISTING_METRIC_FILES and samples_categorization_path.exists():
        print("Already run")
        return
    
    samples_categorization_path.parent.mkdir(parents=True, exist_ok=True)
    samples_categorization = self.calculate_samples_categorization(dataset)
    np.savez_compressed(samples_categorization_path, samples_categorization=samples_categorization)
```

(source: `metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`, lines 48-67)

---

## Available Categorizers

### SAMPLES_CATEGORIZER Enum

**Source:** `resources/data_types.py::SAMPLES_CATEGORIZER`

| ID | Name | Description | Output Shape |
|---:|------|-------------|--------------|
| 1 | `COUNT_WRONG_CLASSIFICATIONS` | Number of times sample is misclassified during AL | `(n_samples,)` |
| 2 | `SWITCHES_CLASS_OFTEN` | Frequency of prediction class changes | `(n_samples,)` |
| 3 | `CLOSENESS_TO_DECISION_BOUNDARY` | Distance to model's decision boundary | `(n_samples,)` |
| 4 | `REGION_DENSITY` | Local sample density in feature space | `(n_samples,)` |
| 5 | `MELTING_POT_REGION` | Degree of class mixing in local neighborhood | `(n_samples,)` |
| 6 | `INCLUDED_IN_OPTIMAL_STRATEGY` | Whether sample is in oracle optimal solution | `(n_samples,)` |
| 7 | `CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN` | k-NN distance to same-class samples | `(n_samples,)` |
| 8 | `CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN` | k-NN distance to different-class samples | `(n_samples,)` |
| 9 | `CLOSENESS_TO_CLUSTER_CENTER` | Distance to assigned cluster centroid | `(n_samples,)` |
| 10 | `IMPROVES_ACCURACY_BY` | Accuracy improvement if sample is labeled | `(n_samples,)` |
| 11 | `AVERAGE_UNCERTAINTY` | Mean model prediction uncertainty | `(n_samples,)` |
| 12 | `OUTLIERNESS` | Isolation forest anomaly score | `(n_samples,)` |
| 13 | `CLOSENESS_TO_SAMPLES_OF_SAME_CLASS` | Mean distance to all same-class samples | `(n_samples,)` |
| 14 | `CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS` | Mean distance to all different-class samples | `(n_samples,)` |

(source: `resources/data_types.py::SAMPLES_CATEGORIZER`, lines 512-528)

---

## Categorizer Details

### COUNT_WRONG_CLASSIFICATIONS

Counts how many times each sample is misclassified across AL cycles and repetitions.

**Implementation:** Aggregates predictions from `y_pred_*.parquet` files across all completed experiments.

**Use case:** Identify consistently hard-to-classify samples.

(source: `metrics/computed/base_samples_categorizer.py::COUNT_WRONG_CLASSIFICATIONS`)

### CLOSENESS_TO_DECISION_BOUNDARY

Measures distance to the model's decision hyperplane.

**Implementation:** Uses model's `decision_function()` when available.

**Use case:** Uncertainty-based strategies often query boundary samples.

(source: `metrics/computed/base_samples_categorizer.py::CLOSENESS_TO_DECISION_BOUNDARY`)

### REGION_DENSITY

Local density computed via k-NN distance aggregation.

**Implementation:** Average distance to k nearest neighbors.

**Use case:** Density-weighted strategies prefer dense regions.

(source: `metrics/computed/base_samples_categorizer.py::REGION_DENSITY`)

### MELTING_POT_REGION

Measures class mixing in local neighborhood.

**Implementation:** Fraction of k-NN neighbors with different class labels.

**Use case:** Identifies boundary regions between classes.

(source: `metrics/computed/base_samples_categorizer.py::MELTING_POT_REGION`)

### OUTLIERNESS

Anomaly score from Isolation Forest.

**Implementation:** `sklearn.ensemble.IsolationForest`

**Use case:** Identify samples that are unlike the training distribution.

(source: `metrics/computed/base_samples_categorizer.py::OUTLIERNESS`)

### CLOSENESS_TO_CLUSTER_CENTER

Distance to assigned cluster centroid.

**Implementation:** K-Means clustering followed by distance computation.

**Use case:** Diversity strategies may prefer cluster representatives.

(source: `metrics/computed/base_samples_categorizer.py::CLOSENESS_TO_CLUSTER_CENTER`)

---

## Output Storage

### File Structure

```
OUTPUT_PATH/<EXP_TITLE>/
├── _COUNT_WRONG_CLASSIFICATIONS/
│   ├── Iris.npz
│   ├── wine_origin.npz
│   └── ...
├── _CLOSENESS_TO_DECISION_BOUNDARY/
│   ├── Iris.npz
│   └── ...
├── _REGION_DENSITY/
│   └── ...
└── ...
```

### File Format

Each `.npz` file contains a single array with key `samples_categorization`:

```python
import numpy as np

# Load categorization
data = np.load("OUTPUT_PATH/exp/_COUNT_WRONG_CLASSIFICATIONS/Iris.npz")
categorization = data["samples_categorization"]
print(categorization.shape)  # (150,) for Iris
print(categorization[:10])   # First 10 samples' wrong classification counts
```

(source: `metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`, lines 64-66)

---

## Producer/Consumer Table

| Producer | Artifact | Consumer | Purpose |
|----------|----------|----------|---------|
| `02_run_experiment.py` | `y_pred_*.parquet` | `COUNT_WRONG_CLASSIFICATIONS` | Prediction aggregation |
| `02_run_experiment.py` | `selected_indices.csv.xz` | `INCLUDED_IN_OPTIMAL_STRATEGY` | Optimal solution comparison |
| Dataset files | `<dataset>.csv` | All categorizers | Feature vectors |
| Dataset files | `<dataset>_distances.csv.gzip.parquet` | Distance-based categorizers | Precomputed distances |
| `03_calculate_dataset_categorizations.py` | `_<CATEGORIZER>/<DATASET>.npz` | Eva scripts | Sample-level analysis |
| `_<CATEGORIZER>/<DATASET>.npz` | Per-sample arrays | `eva_scripts/*.py` | Correlation studies |

(source: `03_calculate_dataset_categorizations.py`, `metrics/computed/base_samples_categorizer.py`)

---

## Using Categorizations in Eva Scripts

### Loading Categorization Data

```python
import numpy as np
from pathlib import Path

def load_categorization(output_path: Path, categorizer_name: str, dataset_name: str) -> np.ndarray:
    """Load per-sample categorization array."""
    path = output_path / f"_{categorizer_name}" / f"{dataset_name}.npz"
    data = np.load(path)
    return data["samples_categorization"]

# Example
categorization = load_categorization(
    Path("/path/to/output/exp"),
    "CLOSENESS_TO_DECISION_BOUNDARY",
    "Iris"
)
```

### Correlating with Query Patterns

```python
import pandas as pd
import numpy as np

# Load selected indices
selected_df = pd.read_csv("OUTPUT_PATH/exp/ALIPY_UNCERTAINTY_LC/Iris/selected_indices.csv.xz")

# Load categorization
boundary_closeness = load_categorization(output_path, "CLOSENESS_TO_DECISION_BOUNDARY", "Iris")

# Analyze which samples were queried
for _, row in selected_df.iterrows():
    queried_indices = eval(row["1"])  # Cycle 1 queries
    queried_closeness = boundary_closeness[queried_indices]
    print(f"Mean boundary closeness of queried samples: {queried_closeness.mean()}")
```

---

## Computation Dependencies

### Required Inputs

| Categorizer | Requires |
|-------------|----------|
| `COUNT_WRONG_CLASSIFICATIONS` | Completed experiments with predictions |
| `SWITCHES_CLASS_OFTEN` | Completed experiments with predictions |
| `CLOSENESS_TO_DECISION_BOUNDARY` | Model with `decision_function()` |
| `REGION_DENSITY` | Dataset feature vectors |
| `MELTING_POT_REGION` | Dataset with labels |
| `CLOSENESS_TO_CLUSTER_CENTER` | Dataset feature vectors |
| `OUTLIERNESS` | Dataset feature vectors |
| `*_kNN` variants | Precomputed distance matrix |

### Distance Matrix

Some categorizers use precomputed pairwise distances:

**Location:** `DATASETS_PATH/<dataset>_distances.csv.gzip.parquet`

**Computed by:** `00_download_datasets.py` (when `DATASETS_COMPUTE_DISTANCES=True`)

(source: `misc/config.py::Config.DATASETS_COMPUTE_DISTANCES`, `misc/config.py::Config.DATASETS_DISTANCES_APPENDIX`)

---

## Idempotency

Categorization computation is idempotent:
- Checks if output file exists before computing
- Skips if already computed (unless `OVERWRITE_EXISTING_METRIC_FILES=True`)

```python
# From base_samples_categorizer.py
if not self.config.OVERWRITE_EXISTING_METRIC_FILES and samples_categorization_path.exists():
    print("Already run")
    return
```

(source: `metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`, lines 53-55)

---

## Cross-References

- **[Architecture](architecture.md)**: System design overview
- **[Data Model](data_model.md)**: Schema definitions
- **[Eva Scripts](eva_scripts.md)**: Scripts that consume categorizations
- **[Pipeline](pipeline.md)**: When to run categorization
