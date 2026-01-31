# Utilities Reference

This document catalogs the utility modules in [`misc/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc) and helper scripts in [`scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts). Each entry indicates whether the utility is pipeline-critical, optional, or a debug/legacy tool.

---

## misc/ Directory

The [`misc/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc) directory contains shared utility modules used throughout OGAL.

### Module Catalog

| Module | Category | Purpose | Safe to Ignore? |
|--------|----------|---------|-----------------|
| `config.py` | **Pipeline-critical** | Central configuration manager | No - required by all scripts |
| `helpers.py` | **Pipeline-critical** | Workload management, data loading | No - used by pipeline and eva_scripts |
| `logging.py` | Pipeline-critical | Logging utilities | No - used for experiment logging |
| `io_utils.py` | Utility | Common I/O functions for results | Yes - convenience module |
| `plotting.py` | Utility | Matplotlib/seaborn styling | Yes - only for visualization |
| `Errors.py` | Utility | Custom exception classes | Yes - only used for error handling |

---

### config.py

**Category:** Pipeline-critical

**Purpose:** Central configuration management for all OGAL scripts.

**Key exports:**
- `Config` class - loads and manages all configuration

**Used by:** Every script in the pipeline

**Details:** See [Configuration](configuration.md)

(source: [`misc/config.py::Config`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py))

---

### helpers.py

**Category:** Pipeline-critical

**Purpose:** Shared helper functions for workload management, data loading, and result processing.

**Key exports:**

| Function | Purpose | Used By |
|----------|---------|---------|
| `append_and_create()` | Append dict to CSV, create if missing | Result writing |
| `append_and_create_manually()` | Append string to file | Logging |
| `get_df()` | Load DataFrame with error handling | All scripts |
| `get_glob_list()` | Find metric files matching pattern | Eva scripts |
| `get_done_workload_joined_with_metric()` | Join workload with metric data | Analysis |
| `create_fingerprint_joined_timeseries_csv_files()` | Create time series parquets | Eva scripts |
| `prepare_eva_pathes()` | Setup eva script output directories | Eva scripts |
| `create_workload()` | Generic workload creation | Post-processing |
| `run_from_workload()` | Execute function across workload | Post-processing |

(source: [`misc/helpers.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py))

---

### logging.py

**Category:** Pipeline-critical

**Purpose:** Simple logging utilities with thread ID and timestamp.

**Key exports:**
- `init_logger(logfilepath)` - Initialize logger with file path
- `log_it(message)` - Log message to console or file

**Usage:**
```python
from misc.logging import init_logger, log_it

init_logger("console")  # or init_logger("/path/to/logfile.log")
log_it("Starting experiment")
```

(source: [`misc/logging.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/logging.py))

---

### io_utils.py

**Category:** Utility

**Purpose:** Common I/O functions for loading/writing OGAL results.

**Key exports:**
- `load_results()` - Load results CSV/parquet
- `write_results()` - Write results with compression
- `compute_run_id()` - Generate unique run identity hash
- `validate_schema()` - Check results schema compliance
- `load_metric_file()` - Load specific metric file
- `get_strategy_dataset_pairs()` - List available results

**Safe to ignore:** Yes - convenience functions that can be replaced with manual pandas calls.

(source: [`misc/io_utils.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/io_utils.py))

---

### plotting.py

**Category:** Utility

**Purpose:** Matplotlib and seaborn styling for publication-quality figures.

**Key exports:**
- `set_seaborn_style()` - Configure seaborn for paper figures
- `set_matplotlib_size()` - Calculate figure dimensions for LaTeX
- `_rename_strategy()` - Human-readable strategy names
- `_rename_learner_model()` - Human-readable model names

**Safe to ignore:** Yes - only needed for generating paper figures.

(source: [`misc/plotting.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/plotting.py))

---

### Errors.py

**Category:** Utility

**Purpose:** Custom exception classes.

**Key exports:**
- `NoStrategyError` - Raised when strategy not found
- `WrongFrameworkError` - Raised when wrong framework adapter used

**Safe to ignore:** Yes - internal error handling.

(source: [`misc/Errors.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/Errors.py))

---

## scripts/ Directory

The [`scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts) directory contains utility scripts for data processing, fixing issues, and analysis.

### Script Categories

| Category | Description |
|----------|-------------|
| **Pipeline-critical** | Required for the main experiment pipeline |
| **Data maintenance** | Fix or transform existing data |
| **Analysis** | Generate insights from results |
| **Debug** | Diagnose issues |
| **Legacy** | Old code, may not work |

---

### Pipeline-Critical Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `validate_results_schema.py` | Validate results before merge | Before adding new results |
| `export_strategy_catalog.py` | Generate strategy docs from code | When updating docs |

---

### Data Maintenance Scripts

#### Workload Management

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `create_dense_workload.py` | Create reduced workload subset | When creating smaller test set |
| `create_new_extended_dense_workload.py` | Extend dense workload | Adding experiments to dense subset |
| `reduce_to_dense.py` | Filter results to dense workload | Before analysis on subset |
| `merge_two_workloads.py` | Combine workloads from different experiments | Merging experiment results |

(source: [`scripts/create_dense_workload.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/create_dense_workload.py), `scripts/merge_two_workloads.py`)

#### Data Conversion

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `convert_dataset_distances_to_parqet.py` | Convert distance CSVs to parquet | Performance optimization |
| `convert_metrics_csvs_to_exp_id_csvs.py` | Reorganize metrics by EXP_ID | Legacy conversion |
| `convert_y_pred_to_parquet.py` | Convert prediction CSVs to parquet | Storage optimization |

(source: [`scripts/convert_*.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/convert_*.py))

#### Data Fixes

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `fix_apply_runtime_limit_post_mortem.py` | Apply timeout limits retroactively | After runtime limit changes |
| `fix_check_if_dupicate_param_combinations_exist.py` | Find duplicate experiments | Diagnosing data issues |
| `fix_duplicate_header_columns.py` | Remove duplicate CSV headers | After CSV corruption |
| `fix_early_stopping_dict_keys_too_small_error.py` | Fix early stopping issues | After specific bug |
| `fix_macro_f1_score_duplicates.py` | Remove duplicate F1 rows | Data cleanup |
| `fix_oom_workload.py` | Fix OOM tracking file | After OOM issues |
| `fix_reduce_number_precision.py` | Reduce float precision | Storage optimization |
| `fix_remove_unnamed_column.py` | Remove pandas unnamed column | After CSV export issues |
| `fix_unconverted_y_parquet.py` | Fix incomplete conversions | After interrupted conversion |
| `merge_duplicate_parquets.py` | Merge duplicate parquet files | After duplicate detection |
| `remove_duplicated_exp_ids.py` | Remove duplicate experiment IDs | Data cleanup |
| `remove_dataset_results.py` | Delete results for specific dataset | When rerunning dataset |
| `remove_lbfgs_mlp_results.py` | Remove LBFGS MLP results | Legacy cleanup |
| `remove_oom_results_from_metric_files.py` | Clean OOM-contaminated results | After OOM recovery |
| `replace_broken_parquet_csvs_with_working_file.py` | Replace corrupted files | After file corruption |

(source: [`scripts/fix_*.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/fix_*.py), `scripts/remove_*.py`)

---

### Analysis Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `check_if_exp_ids_are_present.py` | Verify experiment completeness | Before analysis |
| `find_broken_file.py` | Find corrupted metric files | Diagnosing errors |
| `find_missing_exp_ids_in_metric_files.py` | Find missing experiments | Before rerunning |
| `single_learning_curve.py` | Plot single learning curve | Quick visualization |
| `exp_results_data_format_test.py` | Test data format compliance | Validation |
| `create_auc_selected_ts.py` | Create AUC time series | Analysis preparation |

(source: [`scripts/find_*.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/find_*.py), `scripts/*_test.py`)

---

### Rerun Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `rerun_broken_dataset_categorizations.py` | Recompute failed categorizations | After categorization errors |
| `rerun_broken_experiments.py` | Rerun failed experiments | Recovery from failures |
| `rerun_missing_exp_ids.py` | Rerun missing experiments | Completing partial runs |

(source: [`scripts/rerun_*.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/rerun_*.py))

---

### Data Generation Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `create_gaussian.py` | Generate Gaussian test dataset | Testing |
| `create_xor.py` | Generate XOR test dataset | Testing |

(source: [`scripts/create_gaussian.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/create_gaussian.py), `scripts/create_xor.py`)

---

### Utility Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `render_mermaid.py` | Render Mermaid diagrams | Documentation |

(source: [`scripts/render_mermaid.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/render_mermaid.py))

---

## Safe to Ignore Summary

**Can be safely ignored for basic usage:**

| Category | Files |
|----------|-------|
| Plotting utilities | [`misc/plotting.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/plotting.py) |
| Error classes | [`misc/Errors.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/Errors.py) |
| I/O convenience | [`misc/io_utils.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/io_utils.py) |
| All `fix_*.py` scripts | Unless you encounter the specific issue |
| All `rerun_*.py` scripts | Unless you need recovery |
| Data generation | `create_gaussian.py`, `create_xor.py` |

**Must understand for pipeline usage:**

| Category | Files |
|----------|-------|
| Configuration | [`misc/config.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py) |
| Helpers | [`misc/helpers.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py) |
| Logging | [`misc/logging.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/logging.py) |
| Validation | [`scripts/validate_results_schema.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/validate_results_schema.py) |

---

## Cross-References

- **[Architecture](architecture.md)**: How utilities fit into the system
- **[Configuration](configuration.md)**: Config module details
- **[Eva Scripts](eva_scripts.md)**: Scripts that use these utilities
- **[Data Enrichment](data_enrichment.md)**: Validation script usage
