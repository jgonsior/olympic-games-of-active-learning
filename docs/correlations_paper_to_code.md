# Correlations: Paper Definitions ↔ Code

This page provides canonical definitions of the **three correlation metrics** used in the OGAL paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)), including their mathematical formulations (LaTeX), visual aids, and precise mappings to code implementations.

!!! tip "Related Documentation"
    - **[Evaluation Pipeline](evaluation_pipeline.md)**: Step-by-step guide from raw experiments to final outputs
    - **[Eva Scripts Reference](eva_scripts.md)**: Detailed catalog of all evaluation scripts

---

## Why Three Different Correlations?

OGAL evaluates active learning strategies across multiple dimensions to ensure robust comparisons:

1. **Metric-based correlation (Pearson)**: Do two strategies achieve similar performance values?
2. **Queried samples-based correlation (Jaccard)**: Do two strategies select the same samples during active learning?
3. **Leaderboard ranking invariance (Kendall tau-b)**: Do two evaluation approaches rank strategies in the same order?

Each correlation type answers a fundamentally different research question and uses different input artifacts from the OGAL pipeline.

---

## Summary Table

| Paper Term | Mathematical Definition | What It Measures | OGAL Input Artifact | OGAL Output Artifact | Code Pointer |
|------------|-------------------------|------------------|---------------------|----------------------|--------------|
| **Metric-based correlation (Pearson)** | Pearson's $r$ on paired metric vectors | Linear relationship between performance metrics across configurations | Time series parquets (`_TS/*.parquet`) with metric values per configuration | Correlation matrices (`plots/AUC/*.parquet`, `plots/basic_metrics/*.parquet`) | [`eva_scripts/workload_reduction.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/workload_reduction.py) (scipy.stats.pearsonr) |
| **Queried samples-based correlation (Jaccard)** | Jaccard similarity $J = \frac{\|Q_1 \cap Q_2\|}{\|Q_1 \cup Q_2\|}$ aggregated across AL iterations | Sample selection agreement between strategies | `selected_indices.parquet` with queried sample IDs per iteration | Jaccard similarity heatmaps (`plots/single_hyperparameter/*/single_indice_*.parquet`) | [`eva_scripts/single_hyperparameter_evaluation_indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_hyperparameter_evaluation_indices.py#L260) |
| **Leaderboard ranking invariance (Kendall tau-b)** | Kendall's $\tau_b$ on strategy ranking vectors | Agreement in strategy ordering between evaluation approaches | Rank matrices from leaderboards (`plots/final_leaderboard/rank_*.parquet`) | Kendall tau values and bootstrap distributions | [`eva_scripts/leaderboard_c6_rebuttal.py::kendall_tau_b_from_orders`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_c6_rebuttal.py#L93-L100) |

---

## 1. Metric-based Correlation (Pearson)

### Paper Definition

Metric-based correlation quantifies the linear relationship between two performance metrics (e.g., accuracy vs. F1-score, or full AUC vs. final value). Given two metric vectors $\mathbf{X} = (x_1, x_2, \ldots, x_n)$ and $\mathbf{Y} = (y_1, y_2, \ldots, y_n)$ measured across $n$ configurations, Pearson's correlation coefficient is:

$$
r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}
$$

where $\bar{x}$ and $\bar{y}$ are the sample means. The coefficient $r \in [-1, 1]$ indicates:
- $r = 1$: perfect positive linear correlation
- $r = 0$: no linear correlation
- $r = -1$: perfect negative linear correlation

### Visual Aid

```mermaid
flowchart LR
    A[Metric Vector 1<br/>X = full_auc values] --> C[Pearson r]
    B[Metric Vector 2<br/>Y = final_value] --> C
    C --> D[Correlation coefficient<br/>r ∈ [-1, 1]]
```

### Code Mapping

**Primary implementation:**
- **File:** [`eva_scripts/workload_reduction.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/workload_reduction.py)
- **Function:** Uses `scipy.stats.pearsonr` (imported at line 15)
- **Input:** Time series parquets in `CORRELATION_TS_PATH` (e.g., `_TS/full_auc_weighted_f1-score.parquet`)
- **Output:** Correlation values stored in workload reduction analysis

**Related scripts:**
- [`eva_scripts/basic_metrics_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/basic_metrics_correlation.py): Computes Pearson correlation matrix between standard ML metrics (accuracy, F1, precision, recall)
  - **Output:** `plots/basic_metrics/Standard Metrics.parquet`
- [`eva_scripts/auc_metric_correlation.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/auc_metric_correlation.py): Computes Pearson correlation between AUC-based aggregation metrics
  - **Output:** `plots/AUC/auc_*.parquet`

### How to Run

```bash
# Compute basic metric correlations (accuracy, F1, precision, recall)
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE your_experiment

# Compute AUC aggregation metric correlations
python -m eva_scripts.auc_metric_correlation --EXP_TITLE your_experiment

# Workload reduction analysis (uses Pearson internally)
python -m eva_scripts.workload_reduction --EXP_TITLE your_experiment
```

**Prerequisites:**
- `05_done_workload.csv` must exist
- Time series parquets (`_TS/*.parquet`) generated by [`04_calculate_advanced_metrics.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/04_calculate_advanced_metrics.py)

---

## 2. Queried Samples-based Correlation (Jaccard Similarity)

### Paper Definition

Queried samples-based correlation measures the agreement between two active learning strategies in terms of which samples they select for labeling. For two strategies $S_1$ and $S_2$ at iteration $t$, let $Q_1^{(t)}$ and $Q_2^{(t)}$ be the sets of sample indices queried by each strategy. The Jaccard similarity coefficient at iteration $t$ is:

$$
J(Q_1^{(t)}, Q_2^{(t)}) = \frac{|Q_1^{(t)} \cap Q_2^{(t)}|}{|Q_1^{(t)} \cup Q_2^{(t)}|}
$$

where:
- $|Q_1^{(t)} \cap Q_2^{(t)}|$ is the number of samples selected by both strategies
- $|Q_1^{(t)} \cup Q_2^{(t)}|$ is the total number of unique samples selected by either strategy
- $J \in [0, 1]$, where $J = 1$ means identical sample selection, $J = 0$ means no overlap

**Aggregation across iterations:** To obtain a single similarity score across all $T$ active learning iterations, OGAL computes the mean Jaccard similarity:

$$
J_{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} J(Q_1^{(t)}, Q_2^{(t)})
$$

**Note on terminology:** In OGAL code, Jaccard similarity is sometimes referred to as a "distance" when converted via $d = 1 - J$, but the canonical paper definition uses similarity directly.

### Visual Aid

```mermaid
flowchart TD
    A[AL Iteration 1<br/>Q1={1,3,5}, Q2={1,2,5}] --> D[Jaccard per iteration]
    B[AL Iteration 2<br/>Q1={2,4,6}, Q2={2,4,7}] --> D
    C[AL Iteration T<br/>Q1={...}, Q2={...}] --> D
    D --> E[Mean Jaccard<br/>Javg = mean of all J values]
    E --> F[Similarity ∈ [0, 1]]
```

### Code Mapping

**Primary implementation:**
- **File:** [`eva_scripts/single_hyperparameter_evaluation_indices.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/single_hyperparameter_evaluation_indices.py)
- **Function:** Line 260 computes `jaccard = len(a.intersection(b)) / len(a.union(b))` where `a` and `b` are sets of queried sample indices
- **Aggregation:** Lines 269-280 apply Jaccard calculation across all iterations and aggregate results
- **Input:** `CORRELATION_TS_PATH/selected_indices.parquet` containing arrays of queried sample IDs per iteration
- **Output:** `plots/single_hyperparameter/{TARGET}/single_indice_{TARGET}_{METRIC}_jaccard.parquet`

**Related scripts:**
- [`eva_scripts/similar_strategies.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/similar_strategies.py): Deprecated script that computed Jaccard between strategies (lines 104-149)
  - **Note:** Functionality now integrated into `single_hyperparameter_evaluation_indices.py`

**In code, this is called:** `selected_indices` correlation, Jaccard similarity/distance

### How to Run

```bash
# Compute Jaccard similarity for hyperparameter influence
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE your_experiment
```

**Prerequisites:**
- `selected_indices.parquet` time series in `CORRELATION_TS_PATH`
  - Generated by [`misc/helpers.py::create_fingerprint_joined_timeseries_csv_files`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/misc/helpers.py) with `metric_names=["selected_indices"]`
- Each experiment must have saved `selected_indices.csv.xz` files during execution

**Output interpretation:**
- Green heatmaps in paper figures show Jaccard similarity
- Higher values (closer to 1) indicate strategies select similar samples
- Used to answer: "Do hyperparameter changes affect which samples are queried?"

---

## 3. Leaderboard Ranking Invariance (Kendall tau-b)

### Paper Definition

Leaderboard ranking invariance measures whether different evaluation metrics (e.g., full AUC vs. final value) produce the same strategy rankings. Given a leaderboard rank matrix $R \in \mathbb{R}^{D \times S}$ where $D$ is the number of datasets and $S$ is the number of strategies, each cell $R_{d,s}$ contains the rank of strategy $s$ on dataset $d$.

**Step 1: Compute final ranking vector.** For each strategy, compute its mean rank across all datasets:

$$
\text{FR}_s = \frac{1}{D} \sum_{d=1}^{D} R_{d,s}
$$

Sort strategies by $\text{FR}_s$ (ascending) to obtain a **ranking vector** (ordered list of strategy names).

**Step 2: Compute Kendall tau-b.** Given two ranking vectors from different metrics, Kendall's tau-b measures rank correlation by counting concordant and discordant pairs:

$$
\tau_b = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}
$$

where:
- $n_c$ = number of concordant pairs (same relative order in both rankings)
- $n_d$ = number of discordant pairs (opposite relative order)
- $n_0 = \frac{n(n-1)}{2}$ (total pairs)
- $n_1, n_2$ adjust for ties

The coefficient $\tau_b \in [-1, 1]$ indicates:
- $\tau_b = 1$: perfect agreement (identical rankings)
- $\tau_b = 0$: no agreement
- $\tau_b = -1$: perfect disagreement (reversed rankings)

### Visual Aid

```mermaid
flowchart TD
    A[Leaderboard Matrix R<br/>rows=datasets, cols=strategies] --> B[Aggregate to ranking vector FR<br/>mean rank per strategy]
    B --> C1[Ranking Vector 1<br/>from full AUC]
    B --> C2[Ranking Vector 2<br/>from final value]
    C1 --> D[Kendall tau-b correlation]
    C2 --> D
    D --> E[Rank agreement τb ∈ [-1, 1]]
```

### Code Mapping

**Primary implementation:**
- **File:** [`eva_scripts/leaderboard_c6_rebuttal.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_c6_rebuttal.py)
- **Function:** `kendall_tau_b_from_orders(order_a: list[str], order_b: list[str]) -> float` (lines 93-100)
  - Converts two ordered strategy lists to position vectors
  - Calls `scipy.stats.kendalltau(va, vb, variant="b")`
- **Helper functions:**
  - `load_rank_matrix(path: str)` (line 64): Loads rank parquet files
  - `leaderboard_order_from_matrix(R: pd.DataFrame)` (line 88): Computes mean rank and sorts to get ranking vector
- **Input:** Rank matrices from `plots/final_leaderboard/rank_*.parquet` (one per metric)
- **Output:** Tau values, facet plots, bootstrap distributions

**Related scripts:**
- [`eva_scripts/leaderboard_single_hyperparameter_influence_analyze.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_single_hyperparameter_influence_analyze.py): Computes Kendall tau to measure hyperparameter influence on leaderboard stability
- [`eva_scripts/analyze_leaderboard_rankings.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/analyze_leaderboard_rankings.py): General leaderboard ranking analysis with Kendall tau

**In code, this is called:** Kendall tau-b correlation, leaderboard invariance, ranking agreement

### How to Run

```bash
# Generate leaderboard rankings and compute Kendall tau-b between metrics
python -m eva_scripts.leaderboard_c6_rebuttal
```

**Prerequisites:**
- Rank matrices in `plots/final_leaderboard/rank_*.parquet`
  - Generated by [`eva_scripts/calculate_leaderboard_rankings.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/calculate_leaderboard_rankings.py) or [`eva_scripts/final_leaderboard.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/final_leaderboard.py)
- Each rank matrix must have rows=datasets, columns=strategies, values=ranks

**Output interpretation:**
- High Kendall tau-b (close to 1) means two metrics produce nearly identical strategy rankings
- Used to answer: "Is the full AUC metric redundant with final value?"
- Bootstrap confidence intervals quantify ranking stability

---

## Terminology Cross-Reference

| Paper Term | OGAL Code Alias | File Pattern |
|------------|----------------|--------------|
| Full mean AUC | `full_auc` | `full_auc_*.parquet`, `full_auc_*.csv.xz` |
| Ramp-up AUC | `ramp_up_auc` | `ramp_up_auc_*.parquet` |
| Plateau AUC | `plateau_auc` | `plateau_auc_*.parquet` |
| Final value | `final_value` | `final_value_*.parquet` |
| First 5 iterations | `first_5` | `first_5_*.parquet` |
| Last 5 iterations | `last_5` | `last_5_*.parquet` |
| Queried sample sets | `selected_indices` | `selected_indices.csv.xz`, `selected_indices.parquet` |
| Weighted F1-score | `weighted_f1-score` | `weighted_f1-score.parquet`, `weighted_f1-score.csv.xz` |

---

## Remaining TODO Items

!!! warning "TODO(verify): Paper Figure Extraction"
    If paper PDF or LaTeX sources with figures are located in the repository, extract relevant correlation diagrams and add them to `docs/assets/paper_figures/` with proper citations. Current implementation uses Mermaid diagrams as placeholders.

!!! info "TODO(verify): Exact Paper Equation Numbers"
    The LaTeX equations above are written from general definitions. If the paper uses specific equation numbers (e.g., "Eq. 3"), cross-reference and add those labels here.

!!! info "TODO(verify): Bootstrap Details"
    [`leaderboard_c6_rebuttal.py`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/eva_scripts/leaderboard_c6_rebuttal.py) performs bootstrap resampling for Kendall tau confidence intervals. Add details on bootstrap methodology if described in the paper.

---

## Additional Resources

- **Paper:** [OGAL arXiv:2506.03817](https://arxiv.org/abs/2506.03817)
- **Dataset bundle:** [DOI 10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)
- **GitHub repository:** [jgonsior/olympic-games-of-active-learning](https://github.com/jgonsior/olympic-games-of-active-learning)
- **Documentation:** [https://jgonsior.github.io/olympic-games-of-active-learning/](https://jgonsior.github.io/olympic-games-of-active-learning/)
