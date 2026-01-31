# Utilities & Misc

The `misc/` directory hosts shared helpers used across the pipeline. Categories below separate pipeline-critical utilities from optional helpers.

| file | category | purpose | when to use | safe to ignore? |
| --- | --- | --- | --- | --- |
| `misc/config.py` | Pipeline-critical | Central configuration loader and path resolver (`Config`, `._pathes_magic`) | Always; all scripts instantiate `Config()` | No |
| `misc/helpers.py` | Pipeline-critical | Workload preparation helpers, dataframe joins, time-series creation (`create_fingerprint_joined_timeseries_csv_files`) | When preparing eva workloads or joining metrics | No |
| `misc/logging.py` | Pipeline-critical | Logging utilities used by runners and metrics | Always through pipeline | No |
| `misc/io_utils.py` | Optional helper | File I/O utilities | When custom I/O is needed | Yes, unless extending I/O |
| `misc/plotting.py` | Optional helper | Plot styling helpers (`set_matplotlib_size`, renamers) | When generating plots programmatically | Yes, unless plotting |
| `misc/Errors.py` | Legacy/debug | Error class catalog | Rarely; primarily debugging | Usually |

Legacy note: `analyse_results/` is deprecated; prefer `eva_scripts/` for evaluation.
