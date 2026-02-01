# Reproduce Paper Results

**You're a researcher who wants to understand how the figures, tables, and conclusions in the OGAL paper were computed.**

This guide maps every paper figure to the exact scripts that produce it.

---

## What You'll Accomplish

- Reproduce the main leaderboards and figures from the paper
- Understand the three correlation types used in the paper
- Run the exact evaluation scripts used to generate paper results

---

## Prerequisites

- Completed the [Analyze the Dataset](analyze_dataset.md) setup
- OPARA archive downloaded and extracted

---

## Paper Reference

**Paper:** [arXiv:2506.03817](https://arxiv.org/abs/2506.03817)

---

## Step 1: Generate Base Artifacts

Before reproducing specific figures, generate the required intermediate files:

```bash
# Required for all analyses
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan

# Required for leaderboards
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
```

---

## Step 2: Reproduce Key Results

### Main Leaderboard (Table 1 / Figure 4)

The main leaderboard ranks AL strategies by average performance.

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**Output:** `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

**Verification:**

```python
import pandas as pd
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")
lb = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet")
print("Top 10 strategies (average rank):")
print(lb.mean(axis=0).sort_values().head(10))
```

---

## Step 3: The Three Correlation Types

The paper analyzes hyperparameter sensitivity using three correlation approaches, each shown in a different color.

### Blue: Metric-based Correlation (Pearson)

**Paper Section:** IV-B1 "Metric-based heatmaps"

**What it measures:** Do two hyperparameter values produce similar metric outcomes?

```bash
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan
```

**Mathematical definition:**

$$
r(V_{b_1}(M), V_{b_2}(M))
$$

Where $V_b(M)$ is the vector of aggregated metric values for hyperparameter value $b$.

**Output:** `plots/single_hyperparameter/*/single_*.parquet`

---

### Green: Queried Samples Correlation (Jaccard)

**Paper Section:** IV-B2 "Queried samples-based heatmaps"

**What it measures:** Do two strategies select the same samples?

```bash
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan
```

**Mathematical definition:**

$$
J(\widehat{Q_{b_1}}, \widehat{Q_{b_2}}) = \frac{|\widehat{Q_{b_1}} \cap \widehat{Q_{b_2}}|}{|\widehat{Q_{b_1}} \cup \widehat{Q_{b_2}}|}
$$

Where $\widehat{Q}$ is the union of queried samples across all AL cycles.

**Output:** `plots/single_hyperparameter/*/single_indice_*_jaccard.parquet`

---

### Orange: Leaderboard Ranking Invariance (Kendall tau-b)

**Paper Section:** IV-B3 "Leaderboard ranking invariance-based heatmaps"

**What it measures:** Do different evaluation approaches rank strategies the same way?

```bash
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

**Mathematical definition:**

$$
\tau_b = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}
$$

Where $n_c$ = concordant pairs, $n_d$ = discordant pairs.

**Output:** Kendall tau heatmaps

---

## Step 4: Additional Paper Figures

### Learning Curves (Figure 2)

```bash
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

### Runtime Analysis (Figure 7)

```bash
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

**Output:** `plots/runtime/query_selection_time.parquet`

### Metric Correlations

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.auc_metric_correlation --EXP_TITLE full_exp_jan
```

---

## Step 5: Generate All Paper Plots

To regenerate all publication-ready figures at once:

```bash
python -m eva_scripts.redo_plots_for_paper --EXP_TITLE full_exp_jan
```

---

## Mapping: Paper Figures → Scripts

| Paper Figure | Script | Output |
|--------------|--------|--------|
| Table 1 (Leaderboard) | `final_leaderboard.py` | `plots/final_leaderboard/*.parquet` |
| Figure 2 (Learning curves) | `single_learning_curve_example.py` | `plots/single_learning_curve/*.parquet` |
| Figure 4 (Heatmaps blue) | `single_hyperparameter_evaluation_metric.py` | `plots/single_hyperparameter/*` |
| Figure 5 (Heatmaps green) | `single_hyperparameter_evaluation_indices.py` | `plots/single_hyperparameter/*_jaccard.parquet` |
| Figure 6 (Heatmaps orange) | `leaderboard_single_hyperparameter_influence*.py` | Kendall tau matrices |
| Figure 7 (Runtime) | `runtime.py` | `plots/runtime/*.parquet` |
| Metric correlations | `basic_metrics_correlation.py` | `plots/basic_metrics/*.parquet` |

---

## Terminology Cross-Reference

| Paper Term | Code Alias | File Pattern |
|------------|-----------|--------------|
| Full mean AUC | `full_auc` | `full_auc_*.parquet` |
| Ramp-up AUC | `ramp_up_auc` | `ramp_up_auc_*.parquet` |
| Final value | `final_value` | `final_value_*.parquet` |
| Weighted F1-score | `weighted_f1-score` | `weighted_f1-score.parquet` |

---

## Next Steps

| Goal | Page |
|------|------|
| Deep dive into correlation math | [Correlations: Paper ↔ Code](../reference/correlations_paper_to_code.md) |
| Understand all eva_scripts | [Eva Scripts Catalog](../reference/eva_scripts_catalog.md) |
| Get research ideas | [Research Ideas](research_ideas.md) |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing intermediate files | Run `learning_curve.py` and `calculate_leaderboard_rankings.py` first |
| Different numbers than paper | Ensure you're using the exact `full_exp_jan` archive |
| Memory errors | Process subsets using `--EXP_DATASET` filters |
