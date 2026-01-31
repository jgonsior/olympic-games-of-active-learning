# Architecture Rationale

Design goals:
- HPC-scale parallelism and sharding of millions of runs ([`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) builds the grid; [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) shards by `WORKER_INDEX`).
- Rerunnable after failures with idempotent outputs ([`framework_runners/base_runner.py::AL_Experiment.run_experiment`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners/base_runner.py) appends to `05_done_workload.csv` / `05_failed_workloads.csv`).
- Grid-first workload generation for deterministic slicing ([`01_create_workload.py::_generate_exp_param_grid`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py)).
- Unified code reuse across scripts (shared `Config` and data types in [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)).
- Fast processing with minimal overhead for millions of runs (vectorized workload creation, per-metric CSV append, NPZ caching for categorizations).

```mermaid
flowchart TD
    CFG[Config (.server_access_credentials.cfg + resources/exp_config.yaml)] --> GRID
    GRID[01_create_workload.py grid] --> WL[01_workload.csv]
    WL --> RUN[02_run_experiment.py (sharded by WORKER_INDEX)]
    RUN --> RAW[Raw outputs: metrics CSVs, selected_indices, timing]
    RAW --> CAT[03_calculate_dataset_categorizations.py]
    RAW --> ADV[04_calculate_advanced_metrics.py]
    CAT --> DER[Derived metrics & dataset enums]
    ADV --> DER
    DER --> EVA[eva_scripts/*]
    EVA --> ART[Final artifacts: _TS/*.parquet, plots/*.parquet/.jpg]
```
