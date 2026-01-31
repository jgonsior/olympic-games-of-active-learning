# Eva Scripts Reference

This document provides a comprehensive catalog of all evaluation scripts in [`eva_scripts/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts). These scripts generate analyses, figures, and tables for the research paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)).

!!! tip "Canonical Evaluation Guide"
    For a step-by-step guide to running these scripts in the correct order, see **[Evaluation Pipeline](evaluation_pipeline.md)**.

---

## Script Catalog Overview

| Script | Purpose | Required Inputs | Outputs | Paper Reference |
|--------|---------|-----------------|---------|-----------------|
| `basic_metrics_correlation.py` | Correlation matrix of standard ML metrics | Per-cycle metric CSVs, `05_done_workload.csv` | `plots/basic_metrics/Standard Metrics.parquet` | Metric comparison |
| `auc_metric_correlation.py` | Correlation of AUC-based aggregation metrics | AUC metric files (`full_auc_*.csv.xz`) | `plots/AUC/auc_*.parquet` | Aggregation comparison |
| `calculate_leaderboard_rankings.py` | Strategy rankings across datasets | Time series parquets in `_TS/` | `plots/final_leaderboard/*.parquet`, `plots/leaderboard_invariances/*.csv` | Leaderboard tables |
| `final_leaderboard.py` | Publication-quality final leaderboard heatmaps | `_TS/*.parquet` | `plots/final_leaderboard/rank_*.parquet`, `*.jpg` | Main results |
| `leaderboard_single_hyperparameter_influence.py` | Impact of single hyperparameter on leaderboard | `_TS/*.parquet` | `plots/leaderboard_single_hyperparameter_influence/*.csv` | RQ1 analysis |
| `leaderboard_single_hyperparameter_influence_analyze.py` | Kendall τ analysis of hyperparameter influence | Above CSV files | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` | RQ1 heatmaps |
| `single_hyperparameter_evaluation_metric.py` | Correlation of metrics across hyperparameter values | `_TS/*.parquet` | `plots/single_hyperparameter/*/single_hyper_*.parquet` | Blue heatmaps |
| `single_hyperparameter_evaluation_indices.py` | Jaccard similarity of queried samples | `_TS/selected_indices.parquet` | `plots/single_hyperparameter/*/single_indice_*.parquet` | Green heatmaps |
| `runtime.py` | Strategy runtime analysis | `query_selection_time.csv` files | `plots/runtime/query_selection_time.parquet` | Runtime bar chart |
| `learning_curve.py` | Aggregated learning curves | Per-cycle metric CSVs | `plots/single_learning_curve/*.parquet` | Learning curve figures |
| `single_learning_curve_example.py` | Single dataset learning curve example | `_TS/weighted_f1-score.parquet` | `plots/single_learning_curve/weighted_f1-score.parquet`, `single_exemplary_learning_curve.parquet` | Fig. 2-3 |
| `redo_plots_for_paper.py` | Re-render publication-quality PDFs | All above parquet files | PDFs in [`plots/`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/plots) subdirectories | All paper figures |
| `merge_multiple_plots_single_page.py` | Combine heatmaps into single figures | Individual PDFs | Merged PDFs | Combined figures |

---

## Detailed Script Documentation

### basic_metrics_correlation.py

**Purpose:** Computes Pearson correlation matrix between standard ML metrics (accuracy, F1-score variants, precision, recall) to show metric redundancy.

**Source:** [`eva_scripts/basic_metrics_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/basic_metrics_correlation.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Done workload | `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv` | CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |
| Per-cycle metrics | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/*.csv.xz` | Compressed CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |

#### Outputs Produced

| Output | Path | Format | Consumed By |
|--------|------|--------|-------------|
| Correlation matrix | `OUTPUT_PATH/<EXP_TITLE>/plots/basic_metrics/Standard Metrics.parquet` | Parquet | `redo_plots_for_paper.py` |
| Heatmap image | `OUTPUT_PATH/<EXP_TITLE>/plots/basic_metrics/Standard Metrics.jpg` | JPEG | Direct use |

#### Key Configuration

- Uses `config.OUTPUT_PATH`, `config.CORRELATION_TS_PATH`
- Metrics evaluated: `accuracy`, `weighted_f1-score`, `macro_f1-score`, `weighted_precision`, `weighted_recall`, `macro_precision`, `macro_recall`

(source: [`eva_scripts/basic_metrics_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/basic_metrics_correlation.py#L26-L35), lines 26-35)

#### Typical Invocation

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE your_experiment
```

#### Common Errors

| Error | Cause | Resolution |
|-------|-------|------------|
| `FileNotFoundError: 05_done_workload.csv` | Experiments not completed | Run [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) first |
| `KeyError: 'EXP_UNIQUE_ID'` | Corrupted metric files | Run [`scripts/find_broken_file.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/find_broken_file.py) |

---

### auc_metric_correlation.py

**Purpose:** Computes correlation between different AUC-based aggregation metrics (full_auc, first_5, last_5, ramp_up_auc, plateau_auc, final_value) to determine which aggregations are redundant.

**Source:** [`eva_scripts/auc_metric_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/auc_metric_correlation.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| AUC metrics | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/full_auc_*.csv.xz` | Compressed CSV | [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py) |
| Done workload | `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv` | CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| AUC correlation | `OUTPUT_PATH/<EXP_TITLE>/plots/AUC/auc_weighted_f1-score.parquet` | Parquet |
| Heatmap image | `OUTPUT_PATH/<EXP_TITLE>/plots/AUC/auc_weighted_f1-score.jpg` | JPEG |

#### Key Configuration

- AUC prefixes: `final_value_`, `first_5_`, `full_auc_`, `last_5_`, `ramp_up_auc_`, `plateau_auc_`

(source: [`eva_scripts/auc_metric_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/auc_metric_correlation.py#L42-L51), lines 42-51)

#### Typical Invocation

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```

---

### calculate_leaderboard_rankings.py

**Purpose:** Computes strategy rankings for each dataset using different ranking methods (rank-based, percentage-based, dataset-normalized percentages) and interpolation strategies (zero, remove, average).

**Source:** [`eva_scripts/calculate_leaderboard_rankings.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/calculate_leaderboard_rankings.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Time series parquet | `OUTPUT_PATH/<EXP_TITLE>/_TS/full_auc_weighted_f1-score.parquet` | Parquet | Created by this script or prior eva_scripts |
| Done workload | `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv` | CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Leaderboard rankings | `OUTPUT_PATH/<EXP_TITLE>/plots/final_leaderboard/<rank_type>_<grid_type>_<interpolation>_<metric>.parquet` | Parquet |
| Leaderboard invariances | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_invariances/leaderboard_types.csv` | CSV |

#### Key Configuration

- `rank_or_percentage`: `"rank"`, `"percentages"`, `"dataset_normalized_percentages"`
- `grid_type`: `"sparse"`, `"dense"`
- `interpolation`: `"none"`, `"remove"`, `"zero"`, `"average_of_same_strategy"`

(source: [`eva_scripts/calculate_leaderboard_rankings.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/calculate_leaderboard_rankings.py#L37-L44), lines 37-44)

#### Typical Invocation

```bash
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE your_experiment
```

---

### final_leaderboard.py

**Purpose:** Generates publication-quality final leaderboard heatmaps with strategies ranked by mean rank across all datasets.

**Source:** [`eva_scripts/final_leaderboard.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/final_leaderboard.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Time series parquet | `OUTPUT_PATH/<EXP_TITLE>/_TS/full_auc_weighted_f1-score.parquet` | Parquet | `calculate_leaderboard_rankings.py` |
| Shared fingerprints | `OUTPUT_PATH/<EXP_TITLE>/_TS/final_leaderboard_shared_fingerprints_*.csv` | CSV | Created by this script |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Rank heatmap data | `OUTPUT_PATH/<EXP_TITLE>/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet` | Parquet |
| Rank heatmap image | `OUTPUT_PATH/<EXP_TITLE>/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.jpg` | JPEG |

#### Key Configuration

- Evaluates multiple AUC prefixes: `full_auc_`, `first_5_`, `last_5_`, `ramp_up_auc_`, `plateau_auc_`, `final_value_`
- Default metric: `weighted_f1-score`

(source: [`eva_scripts/final_leaderboard.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/final_leaderboard.py#L63-L70), lines 63-70)

#### Typical Invocation

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE your_experiment
```

---

### leaderboard_single_hyperparameter_influence.py

**Purpose:** Analyzes how changing a single hyperparameter affects strategy rankings. Creates intermediate CSV files for Kendall τ analysis.

**Source:** [`eva_scripts/leaderboard_single_hyperparameter_influence.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_single_hyperparameter_influence.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Time series parquet | `OUTPUT_PATH/<EXP_TITLE>/_TS/full_auc_weighted_f1-score.parquet` | Parquet | Prior eva_scripts |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Hyperparameter rankings | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_single_hyperparameter_influence/<HYPERPARAMETER>.csv` | CSV |

#### Key Configuration

Evaluated hyperparameters:

- `standard_metric`: Different ML metrics (accuracy, F1 variants)
- `EXP_LEARNER_MODEL`: RF, MLP, SVM
- `EXP_BATCH_SIZE`: 1, 5, 10, 20, 50, 100
- `EXP_DATASET`: Individual datasets
- `EXP_TRAIN_TEST_BUCKET_SIZE`: Train/test split indices
- `EXP_START_POINT`: Initial labeled set indices
- `auc_metric`: Different AUC aggregations

(source: [`eva_scripts/leaderboard_single_hyperparameter_influence.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_single_hyperparameter_influence.py#L38-L46), lines 38-46)

#### Typical Invocation

```bash
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE your_experiment
```

---

### leaderboard_single_hyperparameter_influence_analyze.py

**Purpose:** Computes Kendall τ correlation between rankings under different hyperparameter settings to produce the orange heatmaps in the paper.

**Source:** [`eva_scripts/leaderboard_single_hyperparameter_influence_analyze.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_single_hyperparameter_influence_analyze.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Hyperparameter rankings | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_single_hyperparameter_influence/<HYPERPARAMETER>.csv` | CSV | `leaderboard_single_hyperparameter_influence.py` |
| Gold standard rankings | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_single_hyperparameter_influence/standard_metric.csv` | CSV | `leaderboard_single_hyperparameter_influence.py` |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Kendall τ heatmap data | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_single_hyperparameter_influence/<HYPERPARAMETER>_kendall.parquet` | Parquet |
| Kendall τ heatmap image | `OUTPUT_PATH/<EXP_TITLE>/plots/leaderboard_single_hyperparameter_influence/<HYPERPARAMETER>_kendall.jpg` | JPEG |

#### Typical Invocation

```bash
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE your_experiment
```

---

### single_hyperparameter_evaluation_metric.py

**Purpose:** Computes Pearson correlation of metric values (e.g., full_auc_weighted_f1-score) across different values of a single hyperparameter. Produces the blue heatmaps in the paper.

**Source:** [`eva_scripts/single_hyperparameter_evaluation_metric.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_hyperparameter_evaluation_metric.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Time series parquet | `OUTPUT_PATH/<EXP_TITLE>/_TS/full_auc_weighted_f1-score.parquet` | Parquet | Prior eva_scripts |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Metric correlation heatmap | `OUTPUT_PATH/<EXP_TITLE>/plots/single_hyperparameter/<HYPERPARAMETER>/single_hyper_<HYPERPARAMETER>_full_auc_weighted_f1-score.parquet` | Parquet |
| With confidence intervals | `OUTPUT_PATH/<EXP_TITLE>/plots/single_hyperparameter/<HYPERPARAMETER>/single_hyper_<HYPERPARAMETER>_full_auc_weighted_f1-score_cis.parquet` | Parquet |

#### Key Configuration

Targets to evaluate: `EXP_STRATEGY`, `EXP_LEARNER_MODEL`, `EXP_BATCH_SIZE`, `EXP_DATASET`, `EXP_TRAIN_TEST_BUCKET_SIZE`, `EXP_START_POINT`

(source: [`eva_scripts/single_hyperparameter_evaluation_metric.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_hyperparameter_evaluation_metric.py#L49-L57), lines 49-57)

#### Typical Invocation

```bash
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE your_experiment
```

---

### single_hyperparameter_evaluation_indices.py

**Purpose:** Computes Jaccard similarity of queried sample indices across different hyperparameter values. Produces the green heatmaps in the paper showing which strategies query similar samples.

**Source:** [`eva_scripts/single_hyperparameter_evaluation_indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_hyperparameter_evaluation_indices.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Selected indices parquet | `OUTPUT_PATH/<EXP_TITLE>/_TS/selected_indices.parquet` | Parquet | Created by this script from `selected_indices.csv.xz` files |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Jaccard similarity heatmap | `OUTPUT_PATH/<EXP_TITLE>/plots/single_hyperparameter/<HYPERPARAMETER>/single_indice_<HYPERPARAMETER>_full_auc__selected_indices_jaccard.parquet` | Parquet |

#### Typical Invocation

```bash
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE your_experiment
```

---

### runtime.py

**Purpose:** Analyzes and visualizes query selection runtime across all strategies. Produces the runtime bar chart in the paper.

**Source:** [`eva_scripts/runtime.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/runtime.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Query selection time | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/query_selection_time.csv.xz` | Compressed CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |
| Done workload | `OUTPUT_PATH/<EXP_TITLE>/05_done_workload.csv` | CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Runtime data | `OUTPUT_PATH/<EXP_TITLE>/plots/runtime/query_selection_time.parquet` | Parquet |
| Runtime bar chart | `OUTPUT_PATH/<EXP_TITLE>/plots/runtime/query_selection_time.pdf` | PDF |

#### Key Configuration

- Missing strategies are filled with `config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT` (default: 300s)

(source: [`eva_scripts/runtime.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/runtime.py#L136-L142), lines 136-142)

#### Typical Invocation

```bash
python -m eva_scripts.runtime --EXP_TITLE your_experiment
```

---

### learning_curve.py

**Purpose:** Generates aggregated learning curves showing metric progression across AL cycles for all strategies.

**Source:** [`eva_scripts/learning_curve.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/learning_curve.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Per-cycle metrics | `OUTPUT_PATH/<EXP_TITLE>/<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` | Compressed CSV | [`02_run_experiment.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/02_run_experiment.py) |

#### Outputs Produced

| Output | Path | Format |
|--------|------|--------|
| Learning curve data | `OUTPUT_PATH/<EXP_TITLE>/plots/single_learning_curve/*.parquet` | Parquet |

#### Typical Invocation

```bash
python -m eva_scripts.learning_curve --EXP_TITLE your_experiment
```

---

### single_learning_curve_example.py

**Purpose:** Generates a single illustrative learning curve example and an exemplary abstract learning curve (Fig. 2-3 in paper) showing Strategy A/B/C comparison.

**Source:** [`eva_scripts/single_learning_curve_example.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_learning_curve_example.py)

#### Required Inputs

| Input | Path Pattern | Format | Source |
|-------|-------------|--------|--------|
| Weighted F1 time series | `OUTPUT_PATH/<EXP_TITLE>/_TS/weighted_f1-score.parquet` | Parquet | `learning_curve.py` |

#### Outputs Produced

| Output | Path | Format | Description |
|--------|------|--------|-------------|
| Real learning curve | `OUTPUT_PATH/<EXP_TITLE>/plots/single_learning_curve/weighted_f1-score.parquet` | Parquet | Actual strategy comparison |
| Exemplary curve | `OUTPUT_PATH/<EXP_TITLE>/plots/single_learning_curve/single_exemplary_learning_curve.parquet` | Parquet | Abstract illustration |

(source: [`eva_scripts/single_learning_curve_example.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_learning_curve_example.py#L58-L65), lines 58-65)

#### Typical Invocation

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```000000

---

### redo_plots_for_paper.py

**Purpose:** Re-renders all intermediate Parquet files into publication-quality PDFs with correct fonts, colors, and styling for the research paper.

**Source:** [`eva_scripts/redo_plots_for_paper.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/redo_plots_for_paper.py)

#### Required Inputs

All Parquet files from previous eva_scripts:

| Input | Path Pattern |
|-------|-------------|
| Runtime | `plots/runtime/query_selection_time.parquet` |
| Basic metrics | `plots/basic_metrics/Standard Metrics.parquet` |
| AUC correlation | `plots/AUC/auc_weighted_f1-score.parquet` |
| Final leaderboard | `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet` |
| Learning curves | `plots/single_learning_curve/*.parquet` |
| Single hyperparameter | `plots/single_hyperparameter/*/*.parquet` |
| Kendall heatmaps | `plots/leaderboard_single_hyperparameter_influence/*_kendall.parquet` |

(source: [`eva_scripts/redo_plots_for_paper.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/redo_plots_for_paper.py#L35-L75), lines 35-75)

#### Outputs Produced

Publication-ready PDFs with same names as input Parquets, in same directories.

#### Typical Invocation

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```111111

---

### merge_multiple_plots_single_page.py

**Purpose:** Combines related heatmaps (metric, indices, kendall) into single-page figures for the paper.

**Source:** [`eva_scripts/merge_multiple_plots_single_page.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/merge_multiple_plots_single_page.py)

#### Typical Invocation

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```222222

---

## Additional Eva Scripts

### dataset_stats.py

**Purpose:** Computes and displays dataset statistics.

**Source:** [`eva_scripts/dataset_stats.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/dataset_stats.py)

---

### calculate_dataset_dependend_random_ramp_slope.py

**Purpose:** Calculates the dataset-dependent threshold for distinguishing ramp-up vs plateau phase based on random strategy performance.

**Source:** [`eva_scripts/calculate_dataset_dependend_random_ramp_slope.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/calculate_dataset_dependend_random_ramp_slope.py)

#### Outputs Produced

| Output | Path |
|--------|------|
| Threshold CSV | `OUTPUT_PATH/<EXP_TITLE>/_dataset_dependent_random_ramp_plateau_threshold.csv` |

---

### analyze_leaderboard_rankings.py

**Purpose:** Statistical analysis of leaderboard rankings stability.

**Source:** [`eva_scripts/analyze_leaderboard_rankings.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/analyze_leaderboard_rankings.py)

---

### leaderboard_scenarios.py

**Purpose:** Computes leaderboard rankings under different scenario constraints (minimal hyperparameter grids, dataset subsets).

**Source:** [`eva_scripts/leaderboard_scenarios.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_scenarios.py)

#### Key Configuration

Scenarios: `dataset_scenario`, `start_point_scenario`, `adv_start_scenario`, `min_hyper`, `min_hyper2`, `min_hyper_reduction`, `adv_min`, `real_single_scenarios`

---

### workload_reduction.py

**Purpose:** Analyzes how many hyperparameter combinations are needed to achieve stable leaderboard rankings (RQ2 analysis).

**Source:** [`eva_scripts/workload_reduction.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/workload_reduction.py)

---

### similar_strategies.py

**Purpose:** Identifies redundant AL strategies based on queried sample similarity. **Note:** Deprecated, functionality merged into `single_hyperparameter_evaluation_indices.py`.

**Source:** [`eva_scripts/similar_strategies.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/similar_strategies.py)

---

### strategy_framework_correlation.py

**Purpose:** Analyzes correlation between implementations of the same strategy in different frameworks.

**Source:** [`eva_scripts/strateg_framework_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/strateg_framework_correlation.py)

---

### real_world_scenarios_corrs.py / real_world_scenarios_plots.py

**Purpose:** Analysis and visualization for real-world scenario constraints.

**Source:** [`eva_scripts/real_world_scenarios_corrs.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/real_world_scenarios_corrs.py), [`eva_scripts/real_world_scenarios_plots.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/real_world_scenarios_plots.py)

---

## Evaluation Chain Overview

The eva_scripts form a dependency chain. Run them in this order:

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```333333

---

## Time Series Data Format

Most eva_scripts rely on time series Parquet files in `OUTPUT_PATH/<EXP_TITLE>/_TS/`. These are created on-demand by `create_fingerprint_joined_timeseries_csv_files()` in [`misc/helpers.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py).

### Time Series Schema

| Column | Type | Description |
|--------|------|-------------|
| `EXP_DATASET` | int | Dataset enum value |
| `EXP_STRATEGY` | int | AL strategy enum value |
| `EXP_START_POINT` | int | Initial labeled set index |
| `EXP_BATCH_SIZE` | int | Query batch size |
| `EXP_LEARNER_MODEL` | int | Learner model enum value |
| `EXP_TRAIN_TEST_BUCKET_SIZE` | int | Train/test split bucket |
| `ix` | int | AL cycle index |
| `EXP_UNIQUE_ID_ix` | str | Composite key: `{EXP_UNIQUE_ID}_{ix}` |
| `metric_value` | float | Metric value at this cycle |

(source: [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py#L200-L267), lines 200-267)

---

## Troubleshooting

### Missing Parquet Files

If a script reports missing `.parquet` files, the time series haven't been created yet:

```bash
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment
```444444

### Memory Errors

Large experiments may exhaust memory during correlation computation:

1. Use [`scripts/reduce_to_dense.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/scripts/reduce_to_dense.py) to create a smaller subset
2. Run on HPC with sufficient memory allocation

### Slow Parquet Creation

Time series creation is I/O intensive. The scripts:
1. Read all metric CSVs
2. Sort via shell `sort` command
3. Convert to Parquet

This is normal for large experiments and can take hours.
