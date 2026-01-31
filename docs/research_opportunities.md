# Research Opportunities

This page outlines concrete research directions enabled by the archived dataset ([DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)) and the evaluation scripts (`eva_scripts/`). Each item lists required inputs, starter pipelines, and where to place new derived results.

## Post-hoc stopping criteria analysis

- **Required artifacts:** Per-iteration metrics (`<STRATEGY>/<DATASET>/weighted_f1-score.csv`, etc.) and selected indices (`selected_indices.csv`) produced by `02_run_experiment.py` and saved via `metrics/base_metric.py::Base_Metric.save_metrics`.
- **Starter workflow:** Use `eva_scripts/learning_curve.py` or `eva_scripts/single_learning_curve_example.py` to export learning curves (`plots/single_learning_curve/*.parquet`). These scripts read time-series parquets created from per-cycle metrics (`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`).
- **Analysis idea:** Derive candidate stopping points from plateaus, uncertainty trends, or derivative thresholds on the exported curves.
- **Where to add results:** Store new stopping-point annotations as additional Parquet/CSV files alongside `plots/single_learning_curve/` following the existing enrichment protocol.

## Strategy recommendations / meta-learning

- **Required artifacts:** Dataset categorizations (`_/<CATEGORIZER>/<DATASET>.npz` from `03_calculate_dataset_categorizations.py` and `metrics/computed/base_samples_categorizer.py`), per-iteration metrics, and early-iteration slices of AUC summaries (`full_auc_*.csv.xz` from `04_calculate_advanced_metrics.py`).
- **Starter workflow:** Build training features by joining `_TS/*.parquet` fingerprints (from `misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`) with dataset categorizations consumed via `metrics/computed/DATASET_CATEGORIZATION.py`.
- **Leakage-safe splits:** Split by dataset or by dataset clusters (see robustness section) to avoid train/test leakage across similar datasets.
- **Where to add results:** Add new meta-model outputs in a dedicated subfolder under `plots/meta_learning/` following the data enrichment protocol.

## Robustness and dataset similarity

- **Required artifacts:** Auto-computed dataset categorizations (`_/<CATEGORIZER>/<DATASET>.npz`) and their per-strategy counts (`<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz`), plus advanced metrics capturing stability (e.g., `metrics/computed/DISTANCE_METRICS.py`).
- **Starter workflow:** Cluster datasets using the categorization vectors (loaded from NPZ) and analyze strategy consistency across clusters using `_TS/*.parquet` time-series.
- **Similarity notion:** Use the shared categorizations (e.g., hardness, density) computed in `metrics/computed/base_samples_categorizer.py` as feature embeddings for datasets.
- **Where to add results:** Write cluster assignments and robustness summaries to `plots/dataset_similarity/` and `plots/robustness/`.

## Metric-driven tradeoffs

- **Required artifacts:** Advanced metrics outputs (e.g., AUC variants, time-lag, metric drop, distance metrics) produced by `04_calculate_advanced_metrics.py` (`metrics/computed/*`).
- **Starter workflow:** Use the existing AUC and metric-drop files (`<STRATEGY>/<DATASET>/full_auc_*.csv.xz`, `METRIC_DROP.csv.xz`) and load them into `_TS/*.parquet` via `misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`.
- **Analysis idea:** Identify strategies that optimize one metric while degrading another (e.g., runtime vs. accuracy) and visualize Pareto fronts.
- **Where to add results:** Save trade-off surfaces to `plots/metric_tradeoffs/` aligned with current parquet/JPEG conventions.

## Out-of-distribution / failure-mode discovery

- **Required artifacts:** Per-iteration selections (`selected_indices.csv`), per-cycle predictions (`y_pred_train/test.csv.xz.parquet`), and dataset categorizations (NPZ).
- **Starter workflow:** Join selection patterns with categorizations using `metrics/computed/DATASET_CATEGORIZATION.py` outputs to identify regions where strategies collapse (e.g., outliers or rare classes).
- **Analysis idea:** Detect consistent breakdowns of specific strategy families on certain dataset categories; compare against `05_failed_workloads.csv` for failure correlation.
- **Where to add results:** Document failure clusters in `plots/failure_modes/` with accompanying CSV/Parquet for reproducibility.
