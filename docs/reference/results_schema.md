# Results Format

This document describes the output directory structure, file formats, and result schemas produced by OGAL experiments.

---

## Key Columns That Matter

The 10-15 columns you'll use most often:

| Column | In File | Type | Purpose |
|--------|---------|------|---------|
| `EXP_UNIQUE_ID` | All files | int | **Primary key** linking workload → metrics |
| `EXP_DATASET` | Workload | int | Dataset enum value |
| `EXP_STRATEGY` | Workload | int | AL strategy enum value |
| `EXP_LEARNER_MODEL` | Workload | int | RF=1, MLP=2, SVM=3 |
| `EXP_BATCH_SIZE` | Workload | int | Query batch size (1, 5, 10, 20, 50, 100) |
| `EXP_START_POINT` | Workload | int | Initial labeled set index |
| `0`, `1`, ..., `N` | Metrics | float | Per-cycle metric values |
| `value` | Derived | float | Aggregated metric (e.g., AUC) |
| `metric_value` | Time series | float | Value in `_TS/*.parquet` |

---

## Results Overview

| Category | Source | Examples | Location |
|----------|--------|----------|----------|
| **Raw Outputs** | `02_run_experiment.py` | `accuracy.csv`, `selected_indices.csv` | `<STRATEGY>/<DATASET>/` |
| **Derived Artifacts** | `03_*.py`, `04_*.py` | `full_auc_*.csv.xz`, `_TS/*.parquet` | `_TS/`, `plots/` |

---

## Output Directory Structure

```
OUTPUT_PATH/<EXP_TITLE>/
├── 05_done_workload.csv             # ✓ Completed experiments
├── 05_failed_workloads.csv          # ✗ Failed experiments
├── _TS/                              # Time series for eva_scripts
│   ├── full_auc_weighted_f1-score.parquet
│   └── selected_indices.parquet
├── plots/                            # Generated visualizations
└── <STRATEGY>/<DATASET>/             # Per-experiment results
    ├── accuracy.csv                  # Per-cycle accuracy
    ├── weighted_f1-score.csv         # Per-cycle F1
    ├── selected_indices.csv          # Queried samples
    └── full_auc_accuracy.csv.xz      # Aggregated AUC
```

---

## Minimal Example: How Keys Tie Together

### Workload Row (`05_done_workload.csv`)

| EXP_UNIQUE_ID | EXP_DATASET | EXP_STRATEGY | EXP_LEARNER_MODEL | EXP_BATCH_SIZE |
|---------------|-------------|--------------|-------------------|----------------|
| 12345 | 3 | 7 | 1 | 5 |

### Corresponding Metric Row (`accuracy.csv`)

| EXP_UNIQUE_ID | 0 | 1 | 2 | ... | 99 |
|---------------|-----|-----|-----|-----|------|
| 12345 | 0.72 | 0.78 | 0.82 | ... | 0.95 |

### Corresponding Derived Metric (`full_auc_accuracy.csv.xz`)

| EXP_UNIQUE_ID | value |
|---------------|-------|
| 12345 | 0.87 |

---

## Loading Examples

```python
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load completed experiments
done = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv")

# Load metric results
accuracy = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv")

# Join on primary key
merged = done.merge(accuracy, on="EXP_UNIQUE_ID")
```

---

??? info "Full Schema Reference"
    
    ## Workload Schema
    
    | Column | Type | Description |
    |--------|------|-------------|
    | `EXP_UNIQUE_ID` | int | Unique identifier (0-indexed row after shuffle) |
    | `EXP_DATASET` | int | Dataset enum value |
    | `EXP_STRATEGY` | int | AL strategy enum value |
    | `EXP_LEARNER_MODEL` | int | Learner model enum value |
    | `EXP_BATCH_SIZE` | int | Query batch size |
    | `EXP_RANDOM_SEED` | int | Random seed |
    | `EXP_START_POINT` | int | Initial labeled set index |
    | `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split bucket |
    | `EXP_NUM_QUERIES` | int | Number of AL iterations |
    
    ## Per-Cycle Metrics Schema
    
    | Column | Type | Description |
    |--------|------|-------------|
    | `EXP_UNIQUE_ID` | int | Links to workload row |
    | `0`, `1`, ..., `N` | float | Metric value at each AL iteration |
    
    ## Available Metrics
    
    | File | Description |
    |------|-------------|
    | `accuracy.csv` | Classification accuracy |
    | `weighted_f1-score.csv` | Class-weighted F1 score |
    | `macro_f1-score.csv` | Macro-averaged F1 score |
    | `selected_indices.csv` | Queried sample indices per iteration |
    | `query_selection_time.csv` | Time to select query samples (seconds) |
    | `learner_training_time.csv` | Time to retrain model (seconds) |
    
    ## Derived Metrics (from `04_calculate_advanced_metrics.py`)
    
    | File Pattern | Description |
    |--------------|-------------|
    | `full_auc_<metric>.csv.xz` | AUC over entire learning curve |
    | `ramp_up_auc_<metric>.csv.xz` | AUC during initial "ramp-up" phase |
    | `final_value_<metric>.csv.xz` | Final iteration value |
    | `first_5_<metric>.csv.xz` | Average of first 5 iterations |
    | `last_5_<metric>.csv.xz` | Average of last 5 iterations |
    
    ## Dataset Categorizations (from `03_calculate_dataset_categorizations.py`)
    
    | File | Description |
    |------|-------------|
    | `REGION_DENSITY.csv.xz` | Local sample density |
    | `CLOSENESS_TO_DECISION_BOUNDARY.csv.xz` | Distance to decision boundary |
    | `OUTLIERNESS.csv.xz` | Outlier score |
    | `AVERAGE_UNCERTAINTY.csv.xz` | Mean prediction uncertainty |
    
    ## Time Series Parquets (`_TS/`)
    
    Used by eva_scripts for correlation analysis:
    
    | Column | Type | Description |
    |--------|------|-------------|
    | `EXP_DATASET` | int | Dataset enum |
    | `EXP_STRATEGY` | int | Strategy enum |
    | `EXP_BATCH_SIZE` | int | Batch size |
    | `EXP_LEARNER_MODEL` | int | Learner model |
    | `metric_value` | float | The metric value |

---

## Cross-References

- [Eva Scripts Catalog](eva_scripts_catalog.md) — Scripts that consume these files
- [Add Your Results](../add_results.md) — How to contribute new results
- [Architecture](concepts/architecture_in_10_minutes.md) — Data model overview
