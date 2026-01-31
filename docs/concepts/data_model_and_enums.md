# Data Model & Enums

Canonical entities:
- **Run identity**: A row in `01_workload.csv` identified by `EXP_UNIQUE_ID` (generated in [`01_create_workload.py::_generate_exp_param_grid`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py), lines 40-98).
- **Workload item**: Cartesian product of dataset, strategy, learner, batch size, start point, seed (Config `EXP_GRID_*`).
- **Raw result row**: Per-cycle metric rows saved via [`metrics/base_metric.py::Base_Metric.save_metrics`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/base_metric.py) into `<STRATEGY>/<DATASET>/<metric>.csv`.
- **Derived artifact**: Computed metrics (e.g., AUC) saved via `metrics/computed/*` and aggregated time series `_TS/*.parquet` (from [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py)).

Why enums/IDs exist:
- Stable keys for joins across artifacts (`EXP_UNIQUE_ID`).
- Compact storage (int enums) for millions of rows ([`resources/data_types.py::AL_STRATEGY`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py), `LEARNER_MODEL`, `COMPUTED_METRIC`, `SAMPLES_CATEGORIZER`; [`datasets/__init__.py::DATASET`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py)).
- Reproducibility: workload regeneration yields consistent IDs when `INCLUDE_RESULTS_FROM` is used ([`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py), lines 59-98).

Where enums are defined and mapped:
- Strategies, metrics, categorizers: [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) (e.g., `AL_STRATEGY`, `COMPUTED_METRIC`, `SAMPLES_CATEGORIZER`).
- Datasets: dynamically extended from YAML in [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py) (lines 49-60).
- Mappings to implementations: [`resources/data_types.py::al_strategy_to_python_classes_mapping`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py), `samples_categorizer_to_classes_mapping`.

Adding new IDs safely:
- Extend YAML for datasets ([`resources/openml_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/openml_datasets.yaml) or [`resources/kaggle_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/kaggle_datasets.yaml)) or enums in [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py).
- Follow the enrichment protocol in [`docs/data_enrichment.md`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/docs/data_enrichment.md) to add new runs; reuse [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) to regenerate workloads and maintain consistent `EXP_UNIQUE_ID`.
