# Pipeline

End-to-end workflow for running OGAL experiments — from dataset preparation through
result analysis.  See [Reproduce & Run](personas/reproduce_and_run.md) for the full
command-line reference and [Configuration](configuration.md) for all config keys.

---

## Overview

```
resources/exp_config.yaml
        │
        ▼
01_create_workload.py  ──────►  01_workload.csv  (Cartesian product)
                                       │
                                       ▼
                             02_run_experiment.py  (one worker per row)
                                       │
                                       ▼
                              {STRATEGY}/{DATASET}/*.csv   (raw per-cycle metrics)
                                       │
                                 xz compress
                                       │
                                       ▼
                              {STRATEGY}/{DATASET}/*.csv.xz
                                       │
                          ┌────────────┴────────────────┐
                          ▼                             ▼
       03_calculate_dataset_categorizations.py   04_calculate_advanced_metrics.py
                          │                             │
                          ▼                             ▼
               {DATASET}/*.parquet            {STRATEGY}/{DATASET}/*_auc_*.csv.xz
                          │                             │
                          └─────────────┬───────────────┘
                                        ▼
                          scripts/convert_y_pred_to_parquet.py
                          eva_scripts/calculate_dataset_dependend_random_ramp_slope.py
                                        │
                                        ▼
                          eva_scripts/final_leaderboard.py
                                        │
                                        ▼
                                plots/final_leaderboard/*.parquet
```

---

## Step-by-step

### Step 0 — Prepare datasets (one-time)

```bash
python 00_download_datasets.py
```

Downloads datasets from OpenML/Kaggle, generates train/test splits
(`{dataset}_split.csv`), and computes cosine distance matrices.  This step is
only needed if you are regenerating the dataset files from scratch; the OPARA
archive already contains all pre-computed splits.

### Step 1 — Generate workload

```bash
python 01_create_workload.py --EXP_TITLE <name>
```

Reads the named config block from `resources/exp_config.yaml` and produces:

- `{OUTPUT_PATH}/{EXP_TITLE}/01_workload.csv` — one row per experiment (Cartesian
  product of all `EXP_GRID_*` parameters)
- `{OUTPUT_PATH}/{EXP_TITLE}/02_slurm.slurm` — generated SLURM submission script
  (HPC only)

### Step 2 — Run experiments

```bash
# Local — one worker at a time
python 02_run_experiment.py --EXP_TITLE <name> --WORKER_INDEX <i>

# HPC — submit all workers via SLURM
sbatch {OUTPUT_PATH}/{EXP_TITLE}/02_slurm.slurm
```

Each worker:

1. Loads row `WORKER_INDEX` from `01_workload.csv`
2. Loads the dataset and the pre-computed train/test split
3. Initialises the learner model (RF / MLP / SVM / …)
4. Runs the AL loop for `EXP_NUM_QUERIES` iterations:
   - Pre-query metric hooks
   - Strategy selects `EXP_BATCH_SIZE` samples
   - Post-query metric hooks
   - Model retrained with new labels
   - Post-retraining metric hooks
5. Writes per-cycle metric CSVs to
   `{OUTPUT_PATH}/{EXP_TITLE}/{STRATEGY_NAME}/{DATASET_NAME}/`
6. Appends a row to `05_done_workload.csv` (or `05_failed_workloads.csv` on error)

See [Results format](results_format.md) for the output schema.

### Step 2b — Compress raw CSVs

```bash
xz {OUTPUT_PATH}/{EXP_TITLE}/*/*/**.csv
```

Compresses per-cycle CSV files to `.csv.xz` to reduce disk usage (~10× ratio).

### Step 3 — Dataset categorizations

```bash
python 03_calculate_dataset_categorizations.py \
    --EXP_TITLE <name> --SAMPLES_CATEGORIZER _ALL --EVA_MODE local
```

Computes sample-level hardness features for each dataset
(region density, class overlap, etc.) and writes `.parquet` files under
`{OUTPUT_PATH}/{EXP_TITLE}/{DATASET_NAME}/`.

### Step 4 — Advanced metrics

```bash
python 04_calculate_advanced_metrics.py \
    --EXP_TITLE <name> --COMPUTED_METRICS _ALL --EVA_MODE local
```

Derives aggregated metrics from the per-cycle CSVs (AUC variants, distances, etc.)
and writes them as `.csv.xz` files next to the per-cycle files.

### Step 5 — Prerequisite scripts

```bash
python scripts/convert_y_pred_to_parquet.py --EXP_TITLE <name>
python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE <name>
```

Converts `y_pred_*.csv.xz` files to Parquet and computes the dataset-dependent
random baseline slope needed for normalised leaderboard rankings.

### Step 6 — Generate leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE <name>
```

Produces the strategy-×-dataset rank matrix in
`{OUTPUT_PATH}/{EXP_TITLE}/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`.

---

## Key files produced

| File | Produced by | Contents |
|------|-------------|----------|
| `01_workload.csv` | Step 1 | Full experiment queue |
| `05_done_workload.csv` | Step 2 | Successfully completed rows |
| `05_failed_workloads.csv` | Step 2 | Failed rows with error |
| `05_started_oom_workloads.csv` | Step 2 | OOM-killed rows |
| `{STRATEGY}/{DATASET}/*.csv.xz` | Steps 2 + 2b | Per-cycle metric time series |
| `{DATASET}/*.parquet` | Step 3 | Sample-level dataset features |
| `_TS/*.parquet` | Steps 4–6 (auto) | Aggregated time-series parquets |
| `plots/final_leaderboard/*.parquet` | Step 6 | Strategy rank tables |

---

## Cross-references

- [Configuration](configuration.md) — all config keys and their defaults
- [Results format](results_format.md) — per-cycle CSV schema and column definitions
- [Reproduce & Run](personas/reproduce_and_run.md) — full pipeline walkthrough with paper-run commands
