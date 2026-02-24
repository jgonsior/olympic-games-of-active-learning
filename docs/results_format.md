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

## Metric files

Located under `{OUTPUT_PATH}/{EXP_TITLE}/{STRATEGY_NAME}/{DATASET_NAME}/`.

Each metric file is a **`.csv.xz`** (xz-compressed CSV). The orientation is:

- **One row per experiment.**
- An `EXP_UNIQUE_ID` column that links back to the workload CSVs.
- Columns `0`, `1`, `2`, … `N-1` holding the metric value at each AL cycle.

Common metric files:

| File | Description |
|------|-------------|
| `accuracy.csv.xz` | Overall accuracy per cycle |
| `weighted_f1-score.csv.xz` | Weighted-averaged F1 score per cycle |
| `macro_f1-score.csv.xz` | Macro-averaged F1 score per cycle |
| `query_selection_time.csv.xz` | CPU seconds for sample selection per cycle |
| `learner_training_time.csv.xz` | CPU seconds for `model.fit()` per cycle |
| `selected_indices.csv.xz` | Dataset indices queried at each cycle |

Derived metrics produced by `04_calculate_advanced_metrics.py` follow the same
layout.  Typical file-name patterns: `full_auc_{metric}.csv.xz`,
`ramp_up_auc_{metric}.csv.xz`, `plateau_auc_{metric}.csv.xz`,
`final_value_{metric}.csv.xz`.

---

## Reading results in Python

```python
import pandas as pd

RESULTS_DIR = "/path/to/results"          # OUTPUT_PATH from .server_access_credentials.cfg
EXP_TITLE   = "full_exp_jan"

# 1. Load completed-workload metadata
done = pd.read_csv(f"{RESULTS_DIR}/{EXP_TITLE}/05_done_workload.csv")

# 2. Load a metric file (xz-compressed, one row per experiment)
acc = pd.read_csv(
    f"{RESULTS_DIR}/{EXP_TITLE}/ALIPY_RANDOM/Iris/accuracy.csv.xz",
    compression="xz",
)

# 3. Merge metadata with metric on the shared key
merged = done.merge(acc, on="EXP_UNIQUE_ID")
```

---

## How to sanity-check your results directory

After downloading or generating results, verify that the directory looks
reasonable:

- `{EXP_TITLE}/01_workload.csv` and `05_done_workload.csv` exist and are
  non-empty.
- Strategy sub-directories (e.g., `ALIPY_RANDOM/`) contain dataset
  sub-directories, each with `.csv.xz` metric files.
- At least one metric file (e.g., `accuracy.csv.xz`) can be read with
  `pd.read_csv(..., compression="xz")` and contains an `EXP_UNIQUE_ID` column
  plus numbered cycle columns.

---

## Related pages

- [Run a local smoke test](run_local_smoke_test.md) — hands-on first run
- [Use OPARA archive](use_opara_archive.md) — download pre-computed results
- [Pipeline](pipeline.md) — how and when each file is produced
- [Configuration](configuration.md) — config keys that control output paths
- [Evaluation scripts](evaluation_scripts.md) — scripts that consume these files
