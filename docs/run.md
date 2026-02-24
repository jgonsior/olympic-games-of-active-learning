# Run & reproduce

!!! tip "New here?"
    The recommended path for most users is to **analyze the pre-computed
    OPARA archive** — see the [Home page](index.md) for download and setup.
    This page covers running experiments yourself.

---

## Pipeline at a glance

Every OGAL run follows the same stages:

1. **Configure** — define experiment grid in `resources/exp_config.yaml`.
2. **Generate workload** — `01_create_workload.py` expands the grid into one row per experiment (`01_workload.csv`).
3. **Run experiments** — `02_run_experiment.py` executes a single row (one strategy × dataset × seed combination) and writes per-cycle metric CSVs.
4. **Compress** — raw CSVs are compressed to `.csv.xz` (~10× savings).
5. **Compute dataset features** — `03_calculate_dataset_categorizations.py` produces sample-level hardness `.parquet` files.
6. **Derive advanced metrics** — `04_calculate_advanced_metrics.py` computes AUC variants and aggregated stats.
7. **Generate leaderboard** — helper scripts + `eva_scripts.final_leaderboard` produce the final strategy rank tables.

For output file details see [Results & evaluation](results.md).
For configuration keys and the full script reference see [Reference](reference.md).

---

## Local smoke test

A quick sanity check using the built-in `test` config (2 datasets, 4 strategies — finishes in minutes).

```bash
# 1. Install
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# 2. Configure
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# edit .server_access_credentials.cfg → set OUTPUT_PATH and DATASETS_PATH under [LOCAL]

# 3. Download test datasets (OpenML only — no Kaggle token needed for the test config)
python 00_download_datasets.py

# 4. Generate workload & run one experiment
python 01_create_workload.py --EXP_TITLE test
python 02_run_experiment.py  --EXP_TITLE test --WORKER_INDEX 0
```

Verify with `ls "$OUTPUT_PATH/test/"` — you should see `01_workload.csv`, `05_done_workload.csv`, and strategy folders containing per-dataset CSVs.

---

## HPC / parallel execution

For paper-scale runs (~4.6 M experiments), submit to a SLURM cluster:

```bash
python 01_create_workload.py --EXP_TITLE my_run --RUNNING_ENVIRONMENT hpc
sbatch "${OUTPUT_PATH}/my_run/02_slurm.slurm"
```

Each SLURM array task calls `02_run_experiment.py` with a different `WORKER_INDEX`.
After all jobs finish, run post-processing steps 5–7 from the [pipeline overview](#pipeline-at-a-glance) above.
SLURM-specific keys (`SLURM_TIME_LIMIT`, `SLURM_MEMORY`, …) are documented in [Reference](reference.md).

---

## Advanced: running end-to-end from scratch

!!! warning "Experimental"
    The project's recommended path for new users is analyzing the
    [OPARA release](index.md). Running the full pipeline end-to-end from
    scratch — including dataset acquisition, all workload rows, and
    evaluation — currently requires familiarity with the codebase and may
    need code-level adjustments. It is **not** part of the documented
    happy path.

If you still want to proceed, follow the pipeline stages listed [above](#pipeline-at-a-glance) in order, starting from your own `exp_config.yaml` block.
