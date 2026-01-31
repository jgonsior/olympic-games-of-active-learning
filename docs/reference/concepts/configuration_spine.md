# Configuration Spine

OGAL centralizes configuration in [`misc/config.py::Config`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), which loads:
- `.server_access_credentials.cfg` for local/HPC paths and SLURM settings (`Config._load_server_setup_from_file`, lines 317-320).
- [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml) for experiment grids (`Config._load_exp_yaml`).
- CLI args override both (priority 1).
- Workload row overrides runtime parameters when `WORKER_INDEX` is set (`Config._setup_everything`, lines 194-199).

Path resolution for LOCAL vs HPC is handled in `Config._pathes_magic` (lines 204-310), mapping `LOCAL_OUTPUT_PATH`/`HPC_OUTPUT_PATH` and `LOCAL_DATASETS_PATH`/`HPC_DATASETS_PATH`.

## Key config file
- Shared experiment YAML: [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml)
- Credentials and paths: `.server_access_credentials.cfg` (gitignored)

## How scripts load it
- All pipeline scripts instantiate `Config()` (e.g., [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py), [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py), [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py)).
- Helpers use resolved paths and enums (e.g., [`misc/helpers.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py), `metrics/computed/*`).

## LOCAL vs HPC path resolution
- `.server_access_credentials.cfg` defines `[LOCAL]` and `[HPC]` blocks. `RUNNING_ENVIRONMENT` determines which paths populate `OUTPUT_PATH` and `DATASETS_PATH` (`Config._pathes_magic`, lines 204-310).

## Config key table

| config key | used by scripts | affects | code pointer |
| --- | --- | --- | --- |
| `OUTPUT_PATH`, `DATASETS_PATH` | All pipeline scripts | Where results and datasets are read/written | [`misc/config.py::_pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), lines 204-310 |
| `EXP_GRID_*` (dataset, strategy, batch, etc.) | [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) | Grid expansion and workload size | [`01_create_workload.py::_determine_exp_grid_parameters`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py), lines 31-37; `_generate_exp_param_grid`, lines 40-98 |
| `WORKER_INDEX` | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) | Selects workload row to execute | [`misc/config.py::_setup_everything`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), lines 194-199 |
| `OVERWRITE_EXISTING_METRIC_FILES` | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py), metrics/computed | Whether derived metrics are recomputed | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), lines 27-48; [`metrics/computed/base_computed_metric.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_computed_metric.py), lines 30-38 |
| `EVA_MODE` | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) | Switch between workload creation and execution | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), lines 27-48; [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py), lines 25-55 |
| `DATASETS_TRAIN_TEST_SPLIT_APPENDIX` | Dataset loading & categorizers | Train/test split file resolution | [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py), lines 62-73; [`metrics/computed/base_samples_categorizer.py::_get_train_test_splits`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py), lines 113-124 |
| SLURM params (`SLURM_TIME_LIMIT`, `SLURM_NR_THREADS`, etc.) | `misc/helpers.create_workload` via `prepare_eva_pathes` | HPC job scripts | [`misc/helpers.py::create_workload`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py), [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) defaults |
| `INCLUDE_RESULTS_FROM` | [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) | Deduplicates previously run workloads | [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py), lines 59-98 |
