# Architecture in 10 Minutes

A comprehensive overview of OGAL's design, configuration, data model, and utilities—all in one place.

---

## Pipeline Diagram

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

---

## Design Goals

- **HPC-scale parallelism** — Millions of runs sharded by `WORKER_INDEX` ([`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py))
- **Rerunnable after failures** — Idempotent outputs via `05_done_workload.csv` / `05_failed_workloads.csv`
- **Grid-first workload generation** — Deterministic Cartesian product ([`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py))
- **Unified code reuse** — Shared `Config` and data types in [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py)
- **Fast processing** — Vectorized workload creation, per-metric CSV append, NPZ caching
- **Automatic categorizations** — Dataset/sample-level descriptors computed without manual labeling

---

## Quick Reference: "What You Want" → "Where to Look"

| What You Want | File/Script to Touch |
|---------------|---------------------|
| Change experiment grid | [`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml) |
| Change LOCAL/HPC paths | `.server_access_credentials.cfg` |
| Add a new AL strategy | [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) (enum + mapping) |
| Add a new dataset | [`resources/openml_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/openml_datasets.yaml) or [`resources/kaggle_datasets.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/kaggle_datasets.yaml) |
| Add a new metric | Create class in [`metrics/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics) extending `Base_Metric` |
| Understand path resolution | [`misc/config.py::_pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) (lines 204-310) |
| Add a new framework | Create runner in [`framework_runners/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/framework_runners) extending `AL_Experiment` |
| Generate leaderboards | [`eva_scripts/final_leaderboard.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/final_leaderboard.py) |
| Customize plotting | [`misc/plotting.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/plotting.py) |

---

## Configuration

OGAL centralizes configuration in [`misc/config.py::Config`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py), which loads:

1. **`.server_access_credentials.cfg`** — Local/HPC paths and SLURM settings
2. **[`resources/exp_config.yaml`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/exp_config.yaml)** — Experiment grids
3. **CLI args** — Override both (highest priority)
4. **Workload row** — Overrides runtime parameters when `WORKER_INDEX` is set

### Key Config Parameters

| Config Key | Used By | Affects |
|------------|---------|---------|
| `OUTPUT_PATH`, `DATASETS_PATH` | All scripts | Where results and datasets are read/written |
| `EXP_GRID_*` (dataset, strategy, batch) | `01_create_workload.py` | Grid expansion and workload size |
| `WORKER_INDEX` | `02_run_experiment.py` | Selects workload row to execute |
| `OVERWRITE_EXISTING_METRIC_FILES` | `03_*.py`, `04_*.py` | Whether derived metrics are recomputed |
| `EVA_MODE` | `03_*.py`, `04_*.py` | Switch between workload creation and execution |
| SLURM params | `misc/helpers.py` | HPC job scripts |

---

## Data Model & Enums

### Canonical Entities

| Entity | Description | Where Defined |
|--------|-------------|---------------|
| **Run identity** | Row in `01_workload.csv` identified by `EXP_UNIQUE_ID` | [`01_create_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/01_create_workload.py) |
| **Workload item** | Cartesian product of dataset, strategy, learner, batch, seed | Config `EXP_GRID_*` |
| **Raw result row** | Per-cycle metric rows | [`metrics/base_metric.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/base_metric.py) |
| **Derived artifact** | Computed metrics (AUC), time series | `metrics/computed/*`, `_TS/*.parquet` |

### Why Enums Exist

- **Stable join keys** — `EXP_UNIQUE_ID` across all artifacts
- **Compact storage** — Int enums for millions of rows
- **Reproducibility** — Consistent IDs when `INCLUDE_RESULTS_FROM` is used

### Where Enums Are Defined

| Enum | Location |
|------|----------|
| `AL_STRATEGY`, `COMPUTED_METRIC`, `SAMPLES_CATEGORIZER` | [`resources/data_types.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/resources/data_types.py) |
| `DATASET` | [`datasets/__init__.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/datasets/__init__.py) (dynamically extended from YAML) |
| Strategy → implementation mappings | `al_strategy_to_python_classes_mapping` in `data_types.py` |

---

## Dataset Metadata & Auto Enums

### What Gets Computed Automatically

| Categorizer | Purpose |
|-------------|---------|
| `REGION_DENSITY` | Local sample density |
| `CLOSENESS_TO_DECISION_BOUNDARY` | Distance to decision boundary |
| `MELTING_POT_REGION` | Mixed-class neighborhood indicator |
| `OUTLIERNESS` | Outlier score |
| `AVERAGE_UNCERTAINTY` | Mean prediction uncertainty |

### Storage Locations

| Type | Path |
|------|------|
| NPZ per-sample categorizations | `OUTPUT_PATH/<EXP_TITLE>/_/<CATEGORIZER>/<DATASET>.npz` |
| Per-strategy batch counts | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz` |

### Code Pointers

- Workload + execution switch: [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py)
- NPZ write: [`metrics/computed/base_samples_categorizer.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py)
- NPZ → CSV generation: [`metrics/computed/DATASET_CATEGORIZATION.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DATASET_CATEGORIZATION.py)

---

## Utilities Overview

| File | Category | Purpose |
|------|----------|---------|
| [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) | **Pipeline-critical** | Central configuration loader and path resolver |
| [`misc/helpers.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py) | **Pipeline-critical** | Workload prep, dataframe joins, time-series creation |
| [`misc/logging.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/logging.py) | **Pipeline-critical** | Logging utilities |
| [`misc/io_utils.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/io_utils.py) | Optional | File I/O utilities |
| [`misc/plotting.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/plotting.py) | Optional | Plot styling helpers |

!!! note "Legacy Note"
    [`analyse_results/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/analyse_results) is deprecated; prefer [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts) for evaluation.

---

## Cross-References

- [Runbook](../runbook.md) — Running experiments
- [Eva Scripts](../eva_scripts_catalog.md) — Evaluation scripts
- [Results Schema](../results_schema.md) — Output file formats
- [Strategy Catalog](../strategy_catalog.md) — All AL strategies
