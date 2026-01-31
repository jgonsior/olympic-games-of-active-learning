# Dataset Value

## Compute investment

The full hyperparameter grid required **~3.6 million CPU hours** to execute (paper: [arXiv:2506.03817](https://arxiv.org/abs/2506.03817)). Reusing the archived dataset avoids this cost for downstream analyses.

## Why this matters

- Enables post-hoc analyses without rerunning experiments (raw per-iteration metrics and selections are already computed).
- Enables method development and meta-analyses using a unified schema of strategies, datasets, learners, and seeds.
- Enables strategy recommendations and failure-mode studies by leveraging derived metrics and auto-computed dataset categorizations.

## Dataset inventory

Artifacts are stored under `OUTPUT_PATH/<EXP_TITLE>/` (source: [`misc/config.py::_pathes_magic`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/config.py#L204-L310), lines 204-310). Formats and producers/consumers are traced to the code paths listed.

| artifact | produced by | consumed by | location pattern | format | notes |
| --- | --- | --- | --- | --- | --- |
| Workload definitions | `01_create_workload.py::_generate_exp_param_grid` | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) | `OUTPUT_PATH/<EXP_TITLE>/01_workload.csv` | CSV | Enumerates grid with `EXP_UNIQUE_ID`. |
| Completed workload log | `framework_runners/base_runner.py::AL_Experiment.run_experiment` | `eva_scripts/*`, advanced metrics | `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv` | CSV | Tracks successful runs. |
| Failed/oom workload logs | `framework_runners/base_runner.py::AL_Experiment.run_experiment` | Debugging, reruns | `05_failed_workloads.csv`, `05_started_oom_workloads.csv` | CSV | Error types and OOM tracking. |
| Per-iteration standard metrics | `metrics/Standard_ML_Metrics` via `Base_Metric.save_metrics` | `eva_scripts/*`, advanced metrics | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/weighted_f1-score.csv` (and other metrics) | CSV | Time series per AL cycle. |
| Selected indices | `metrics/Selected_Indices` via `Base_Metric.save_metrics` | [`metrics/computed/DATASET_CATEGORIZATION.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DATASET_CATEGORIZATION.py), eva scripts | `<STRATEGY>/<DATASET>/selected_indices.csv` | CSV | Queried sample indices per cycle. |
| Timing metrics | `metrics/Timing_Metrics` | Runtime analyses | `<STRATEGY>/<DATASET>/query_selection_time.csv`, `learner_training_time.csv` | CSV | Per-iteration timings. |
| Predictions per cycle | `metrics/Predicted_Samples` | `metrics/computed.base_samples_categorizer` | `<STRATEGY>/<DATASET>/y_pred_train.csv.xz.parquet`, `y_pred_test.csv.xz.parquet` | Parquet | Per-cycle predictions (train/test). |
| Advanced metrics (AUC, etc.) | [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) using `metrics/computed/*` | [`eva_scripts/auc_metric_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/auc_metric_correlation.py), leaderboards | `<STRATEGY>/<DATASET>/full_auc_*.csv.xz` | CSV | Derived from per-cycle metrics. |
| Dataset categorizations | [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py) using [`metrics/computed/base_samples_categorizer.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py) | [`metrics/computed/DATASET_CATEGORIZATION.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DATASET_CATEGORIZATION.py), downstream analyses | `_/<CATEGORIZER>/<DATASET>.npz` and `<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz` | NPZ (per-sample), CSV (per-batch counts) | Auto-computed enums per sample. |
| Time-series fingerprints | `misc/helpers.py::create_fingerprint_joined_timeseries_csv_files` | `eva_scripts/*` | `_TS/*.parquet` | Parquet | Joined AUC/time-series fingerprints. |
| Plots / leaderboards | `eva_scripts/*` | Publication artifacts | `plots/**` | Parquet, CSV, JPG | Final figures/tables. |

## What the paper analyzed vs what else exists

- Paper analyses (see `eva_scripts/final_leaderboard.py`, `eva_scripts/leaderboard_single_hyperparameter_influence.py`, [`eva_scripts/auc_metric_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/auc_metric_correlation.py)):
  - Strategy leaderboards across datasets (`plots/final_leaderboard/*.parquet`).
  - Hyperparameter influence on rankings (`plots/leaderboard_single_hyperparameter_influence/*.csv`).
  - Correlation of AUC aggregations (`plots/AUC/*.parquet`).
  - Learning curve summaries (`plots/single_learning_curve/*.parquet`).

- Additional available artifacts not fully exploited in the paper:
  - Per-iteration selected indices for every run (`<STRATEGY>/<DATASET>/selected_indices.csv`) — enables stopping-criteria and sampling-pattern analyses (source: [`metrics/Selected_Indices`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Selected_Indices)).
  - Per-cycle predictions (`y_pred_train/test.csv.xz.parquet`) enabling agreement/disagreement and calibration studies (source: [`metrics/Predicted_Samples`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/Predicted_Samples)).
  - Auto-computed dataset categorizations (`_/<CATEGORIZER>/<DATASET>.npz`, consumption via `metrics/computed/DATASET_CATEGORIZATION.py`) — supports dataset similarity and failure-mode clustering.
  - Advanced metrics beyond final AUC: time-lag, metric drop, distance metrics (`metrics/computed/*`, produced via `04_calculate_advanced_metrics.py`) for trade-off analyses.
  - Runtime breakdowns (`query_selection_time.csv`, `learner_training_time.csv`) for efficiency/throughput studies.

All artifact paths and producers are derived from the code references above; any unverified claims should be annotated as TODO(verify) before publication.
