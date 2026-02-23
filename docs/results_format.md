# Results Format

!!! abstract "When do I need this?"
    Read this page to understand the **schema of every output file** — column
    names, types, and where each file lives on disk.  For a hands-on first run,
    see [Run a local smoke test](run_local_smoke_test.md).  For pre-computed
    results, see [Use OPARA archive](use_opara_archive.md).

All files described below are written by `02_run_experiment.py`
(via `metrics/base_metric.py`) and consumed by the post-processing and
evaluation scripts.  See [Pipeline](pipeline.md) for how these files fit into
the overall workflow.

---

## Workload tracking files

Located directly under `{OUTPUT_PATH}/{EXP_TITLE}/`.

### `01_workload.csv`

One row per experiment, produced by `01_create_workload.py`.

| Column | Type | Description |
|--------|------|-------------|
| `EXP_UNIQUE_ID` | `int` | Auto-assigned integer that uniquely identifies this row |
| `EXP_DATASET` | `int` | `DATASET` enum integer ID |
| `EXP_STRATEGY` | `int` | `AL_STRATEGY` enum integer ID |
| `EXP_LEARNER_MODEL` | `int` | `LEARNER_MODEL` enum integer ID |
| `EXP_BATCH_SIZE` | `int` | Samples queried per AL cycle |
| `EXP_NUM_QUERIES` | `int` | Total AL cycles to run |
| `EXP_START_POINT` | `int` | Start-set index (0-indexed) |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | `int` | Train/test split index (0-indexed) |
| `EXP_RANDOM_SEED` | `int` | Per-experiment random seed |

### `05_done_workload.csv`

Same columns as `01_workload.csv`.  One row is **appended** when a worker
completes successfully.  Re-running `01_create_workload.py` automatically
excludes rows that already appear here.

### `05_failed_workloads.csv`

Same columns as `01_workload.csv`, plus an `error` column containing the
exception type.  One row is appended when a worker raises an unhandled
exception.

### `05_started_oom_workloads.csv`

Same columns as `01_workload.csv`.  A row is written when a worker starts;
if the worker is killed by OOM before appending to `05_done_workload.csv`,
this file retains the orphaned row for diagnosis.

---

## Per-cycle metric files

Located under `{OUTPUT_PATH}/{EXP_TITLE}/{STRATEGY_NAME}/{DATASET_NAME}/`.

Each file stores one value per AL cycle (row index = AL cycle number, 0-based).
Files are written as plain `.csv` by `02_run_experiment.py` and then compressed
to `.csv.xz` (use `xz` or `pandas.read_csv(..., compression='xz')`).

### `Standard_ML_Metrics` — classification performance

Produced by `metrics/Standard_ML_Metrics.py` using scikit-learn's
`classification_report`.

| File | Rows | Description |
|------|------|-------------|
| `accuracy.csv.xz` | `EXP_NUM_QUERIES` | Overall accuracy per cycle |
| `macro_f1-score.csv.xz` | `EXP_NUM_QUERIES` | Macro-averaged F1 score per cycle |
| `weighted_f1-score.csv.xz` | `EXP_NUM_QUERIES` | Weighted-averaged F1 score per cycle |
| `macro_precision.csv.xz` | `EXP_NUM_QUERIES` | Macro-averaged precision per cycle |
| `weighted_precision.csv.xz` | `EXP_NUM_QUERIES` | Weighted-averaged precision per cycle |
| `macro_recall.csv.xz` | `EXP_NUM_QUERIES` | Macro-averaged recall per cycle |
| `weighted_recall.csv.xz` | `EXP_NUM_QUERIES` | Weighted-averaged recall per cycle |

### `Timing_Metrics` — wall-clock timing

Produced by `metrics/Timing_Metrics.py` using `time.process_time()`.

| File | Rows | Description |
|------|------|-------------|
| `query_selection_time.csv.xz` | `EXP_NUM_QUERIES` | CPU seconds for the AL strategy's sample selection per cycle |
| `learner_training_time.csv.xz` | `EXP_NUM_QUERIES` | CPU seconds for `model.fit()` per cycle |

### `Selected_Indices` — queried sample indices

Produced by `metrics/Selected_Indices.py`.

| File | Rows | Description |
|------|------|-------------|
| `selected_indices.csv.xz` | `EXP_NUM_QUERIES` | Global dataset indices of the samples queried at each cycle |

Each row is a list of `EXP_BATCH_SIZE` integers.

### `Predicted_Samples` — model predictions per cycle

Produced by `metrics/Predicted_Samples.py`.

| File | Rows | Description |
|------|------|-------------|
| `y_pred_train.csv.xz` | `EXP_NUM_QUERIES` | Model predictions on the full training set after each retraining |
| `y_pred_test.csv.xz` | `EXP_NUM_QUERIES` | Model predictions on the test set after each retraining |

---

## Derived metric files (Step 4)

Produced by `04_calculate_advanced_metrics.py`, written alongside the per-cycle
files in `{OUTPUT_PATH}/{EXP_TITLE}/{STRATEGY_NAME}/{DATASET_NAME}/`.

| File pattern | Description |
|-------------|-------------|
| `full_auc_{metric}.csv.xz` | Area under the learning curve for the full run |
| `ramp_up_auc_{metric}.csv.xz` | AUC of the initial ramp-up phase |
| `plateau_auc_{metric}.csv.xz` | AUC of the plateau phase |
| `final_value_{metric}.csv.xz` | Final metric value at the last AL cycle |

Where `{metric}` is typically `weighted_f1-score`.

---

## Aggregated parquet files (auto-generated)

Located under `{OUTPUT_PATH}/{EXP_TITLE}/_TS/`.  These files are generated
automatically by evaluation scripts the first time they are needed.

| File | Description |
|------|-------------|
| `full_auc_weighted_f1-score.parquet` | AUC values for all experiments, all datasets |
| `selected_indices.parquet` | Merged selected-indices data across all experiments |
| *(others)* | Additional metrics as generated by `eva_scripts/` |

---

## Reading results in Python

```python
import pandas as pd

RESULTS_DIR = "/path/to/results"   # OUTPUT_PATH from [LOCAL] in .server_access_credentials.cfg

# Load completed workload
done = pd.read_csv(f"{RESULTS_DIR}/full_exp_jan/05_done_workload.csv")

# Load per-cycle metric for one strategy/dataset combination
accuracy = pd.read_csv(
    f"{RESULTS_DIR}/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv.xz",
    compression='xz',
    header=None,
    names=["accuracy"],
)

# Load aggregated time series
ts = pd.read_parquet(f"{RESULTS_DIR}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")
```

---

## Related pages

- [Run a local smoke test](run_local_smoke_test.md) — hands-on first run
- [Use OPARA archive](use_opara_archive.md) — download pre-computed results
- [Pipeline](pipeline.md) — how and when each file is produced
- [Configuration](configuration.md) — config keys that control output paths
- [Evaluation scripts](evaluation_scripts.md) — scripts that consume these files
