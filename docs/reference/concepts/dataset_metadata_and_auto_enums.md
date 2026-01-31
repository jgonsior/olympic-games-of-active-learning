# Dataset Metadata & Auto Enums

Why automatic categorizations:
- Provide dataset-level and sample-level descriptors (hardness, density, boundary proximity) without manual labeling.
- Support downstream grouping, robustness, and similarity analyses at scale.

Where computed:
- Workload creation: [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py) generates workloads for categorizers and runs them.
- Per-sample computation: [`metrics/computed/base_samples_categorizer.py::Base_Samples_Categorizer.categorize_samples`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py) saves NPZ files `_/<CATEGORIZER>/<DATASET>.npz`.
- Per-batch aggregation: [`metrics/computed/DATASET_CATEGORIZATION.py::DATASET_CATEGORIZATION.compute_metrics`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DATASET_CATEGORIZATION.py) loads NPZ categorizations and produces `<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz`.

Storage locations:
- NPZ per-sample categorizations: `OUTPUT_PATH/<EXP_TITLE>/_/<CATEGORIZER>/<DATASET>.npz` (saved in `Base_Samples_Categorizer.categorize_samples`, lines 48-67).
- Per-strategy batch counts: `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/<CATEGORIZER>.csv.xz` (written by `DATASET_CATEGORIZATION.compute_metrics`, lines 59-73).

Downstream usage:
- eva scripts can join categorizations with `_TS/*.parquet` time-series (via [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py)) to correlate dataset traits with strategy performance.
- Advanced metrics ([`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py)) can be extended to incorporate categorizations for meta-features.

Code pointers:
- Workload + execution switch: [`03_calculate_dataset_categorizations.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/03_calculate_dataset_categorizations.py), lines 27-59.
- NPZ write path: [`metrics/computed/base_samples_categorizer.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/base_samples_categorizer.py), lines 48-67.
- NPZ consumption and CSV generation: [`metrics/computed/DATASET_CATEGORIZATION.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/metrics/computed/DATASET_CATEGORIZATION.py), lines 17-73.
