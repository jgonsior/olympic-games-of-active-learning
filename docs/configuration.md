# Configuration Reference

All configuration is loaded by `misc/config.py` from multiple sources in this
priority order (highest wins):

1. **CLI arguments** — `--KEY value` on any script invocation
2. **`.server_access_credentials.cfg`** — INI file with `[LOCAL]` and `[HPC]`
   sections; copy from `.server_access_credentials.cfg.example`
3. **`resources/exp_config.yaml`** — named experiment grids (`EXP_GRID_*` keys)
4. **Workload row** — when `WORKER_INDEX` is set, the matching row from
   `01_workload.csv` is loaded and overwrites the above for per-experiment keys
5. **Built-in defaults** — values shown in the tables below

---

## Path keys (`.server_access_credentials.cfg`)

These keys **must** be present for every run; no built-in default exists.

### `[LOCAL]` section — required for local and laptop runs

| Key | Description |
|-----|-------------|
| `OUTPUT_PATH` | Absolute path where experiment result directories are written |
| `DATASETS_PATH` | Absolute path to the directory containing preprocessed dataset CSV files |
| `CODE_PATH` | *(Optional)* Absolute path to the repository root |

### `[HPC]` section — required only when `RUNNING_ENVIRONMENT=hpc`

| Key | Description |
|-----|-------------|
| `SSH_LOGIN` | SSH login for the cluster head node (e.g. `user@login.cluster.edu`) |
| `WS_PATH` | Absolute path to the workspace directory on the cluster file system |
| `PYTHON_PATH` | Absolute path to the Python interpreter in the conda/poetry environment on HPC |
| `OUTPUT_PATH` | Result directory on the cluster (may differ from `[LOCAL]`) |
| `DATASETS_PATH` | Dataset directory on the cluster (may differ from `[LOCAL]`) |
| `SLURM_PROJECT` | SLURM account name / project allocation |
| `SLURM_MAIL` | Email address for SLURM job notifications |

---

## Runtime keys (CLI or `.server_access_credentials.cfg`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `RUNNING_ENVIRONMENT` | `"local"` \| `"hpc"` | `"local"` | Selects which path block (`[LOCAL]` or `[HPC]`) is used |
| `RANDOM_SEED` | `int` | `1312` | Global NumPy + Python random seed set at process start; pass `-1` to disable |
| `N_JOBS` | `int` | `1` | Number of parallel workers for post-processing steps |
| `EXP_TITLE` | `str` | `"all_strategies_all_datasets_single_random_seed"` | Name of the experiment; selects the block in `exp_config.yaml` and sets output subdirectory |
| `WORKER_INDEX` | `int` | *(none)* | When set, loads row `WORKER_INDEX` from `01_workload.csv` |

---

## Experiment grid keys (`resources/exp_config.yaml`)

These keys define the Cartesian product of hyperparameters.  `01_create_workload.py`
expands them into one row per combination and writes `01_workload.csv`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `EXP_GRID_DATASET` | `List[DATASET]` | *(required)* | Dataset enum names or integer IDs to include |
| `EXP_GRID_STRATEGY` | `List[AL_STRATEGY]` | *(required)* | AL strategy enum names or IDs |
| `EXP_GRID_LEARNER_MODEL` | `List[LEARNER_MODEL]` | `[RF]` | Learner model enum names or IDs |
| `EXP_GRID_BATCH_SIZE` | `List[int]` | `[5]` | Number of samples queried per AL cycle |
| `EXP_GRID_NUM_QUERIES` | `List[int]` | `[0]` | Number of AL cycles; `0` means run until exhaustion |
| `EXP_GRID_START_POINT` | `List[int]` | *(required)* | Pre-generated start-set index (0-indexed) |
| `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` | `List[int]` | `[0,1,2,3,4]` | Pre-generated train/test split index (0-indexed) |
| `EXP_GRID_RANDOM_SEED` | `List[int]` | *(required)* | Per-experiment random seed |
| `METRICS` | `List[str]` | *(required)* | Metric classes to record (e.g. `Standard_ML_Metrics`) |

Ranges are supported with `[start-end]` syntax (e.g. `EXP_GRID_START_POINT: [0-19]`).

---

## SLURM keys (HPC only)

| Key | Default | Description |
|-----|---------|-------------|
| `SLURM_TIME_LIMIT` | `"1:59:59"` | Wall-clock time limit per job |
| `SLURM_NR_THREADS` | `1` | Number of CPU threads per SLURM job |
| `SLURM_MEMORY` | `2210` | Memory per job in MB |
| `SLURM_JOBS_PR_THREAD` | `10` | Number of experiment iterations per thread |
| `SLURM_OFFSET` | `0` | Starting job index offset |
| `SLURM_ITERATIONS_PER_BATCH` | `100` | Batch size for chain-job submission |

---

## Dataset pipeline keys

These keys control how datasets are prepared and are rarely changed.

| Key | Default | Description |
|-----|---------|-------------|
| `DATASETS_AMOUNT_OF_SPLITS` | `5` | Number of train/test split buckets to generate |
| `DATASETS_TEST_SIZE_PERCENTAGE` | `0.4` | Fraction of data held out for testing |
| `DATASETS_COMPUTE_DISTANCES` | `True` | Whether to compute cosine distance matrices |
| `AMOUNT_OF_START_POINTS_TO_GENERATE` | `10000` | Number of candidate start sets generated |

---

## Minimal example

```ini
# .server_access_credentials.cfg
[LOCAL]
OUTPUT_PATH   = /home/user/ogal_results
DATASETS_PATH = /home/user/ogal_datasets
```

```yaml
# resources/exp_config.yaml  (add a new block)
my_run:
  EXP_GRID_DATASET: [Iris, wine_origin]
  EXP_GRID_STRATEGY: [ALIPY_RANDOM]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_NUM_QUERIES: [10]
  EXP_GRID_BATCH_SIZE: [5]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
  EXP_GRID_START_POINT: [0]
  METRICS: [Standard_ML_Metrics]
```

```bash
python 01_create_workload.py --EXP_TITLE my_run
python 02_run_experiment.py  --EXP_TITLE my_run --WORKER_INDEX 0
```

---

## Cross-references

- [Pipeline](pipeline.md) — step-by-step workflow
- [Results format](results_format.md) — output file schema
- [Reproduce & Run](personas/reproduce_and_run.md) — full walkthrough
- [Extend the Benchmark](personas/extend_benchmark.md) — adding strategies / datasets
