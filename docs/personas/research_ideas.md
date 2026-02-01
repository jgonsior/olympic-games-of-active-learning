# Research Ideas

**You're a researcher looking for open questions, unexplored corners of the dataset, and potential research directions.**

This page offers concrete research ideas that can be explored using the OGAL dataset.

---

## Why OGAL Is Unique for Research

The OGAL dataset offers research opportunities that are hard to find elsewhere:

| Feature | Research Opportunity |
|---------|---------------------|
| **4.6M experiments** | Statistical power for robust conclusions |
| **28 strategies × 92 datasets** | Cross-dataset generalization studies |
| **Per-cycle metrics** | Temporal analysis of AL behavior |
| **Queried sample indices** | Sample selection pattern analysis |
| **Multiple learner models** | Learner-strategy interaction studies |
| **Dataset categorizations** | Meta-learning feature space |

---

## Research Idea 1: Stopping Point Analysis

**Question:** When should Active Learning stop? Can we detect performance plateaus early?

### Background

Most AL research focuses on *what* to query, not *when* to stop. The OGAL dataset provides per-cycle metrics that enable fine-grained plateau detection.

### Approach

1. **Load per-cycle metrics:**
   ```python
   import pandas as pd
   ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/weighted_f1-score.parquet")
   ```

2. **Detect plateau onset:**
   - Compute rolling derivative of learning curves
   - Identify when derivative drops below threshold
   - Compare stopping points across strategies

3. **Predict stopping point:**
   - Train model to predict plateau iteration from early iterations
   - Evaluate: Can first 5 iterations predict plateau?

### Key Files

| File | Content |
|------|---------|
| `<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` | Per-cycle F1 scores |
| `_TS/weighted_f1-score.parquet` | Aggregated time series |

### Related Eva Script

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

---

## Research Idea 2: Strategy Recommendation (Meta-Learning)

**Question:** Can we predict which AL strategy will work best for a new dataset?

### Background

No single AL strategy dominates all datasets. A meta-learner could recommend strategies based on dataset characteristics.

### Approach

1. **Extract dataset features:**
   ```python
   import numpy as np
   from pathlib import Path
   
   cats = Path(f"{OGAL_OUTPUT}/full_exp_jan/_/REGION_DENSITY/")
   for npz_file in cats.glob("*.npz"):
       data = np.load(npz_file)
       # Extract statistics: mean, std, skew, etc.
   ```

2. **Build training data:**
   - X: Dataset features (from categorizations)
   - Y: Best-performing strategy for that dataset

3. **Train meta-model:**
   - Use cross-validation with dataset-level splits (no leakage)
   - Compare meta-learner to random/majority baseline

### Key Files

| File | Content |
|------|---------|
| `_/<CATEGORIZER>/<DATASET>.npz` | Per-sample categorizations |
| `_TS/full_auc_weighted_f1-score.parquet` | Strategy performance |

### Open Questions

- Which dataset features predict strategy success?
- Can early AL iterations serve as cheap proxy features?
- How does the meta-model generalize to truly new datasets?

---

## Research Idea 3: Sample Selection Patterns

**Question:** Do different strategies select fundamentally different samples? When does selection strategy matter most?

### Background

OGAL records which samples each strategy queries. This enables analysis of selection patterns beyond performance metrics.

### Approach

1. **Load selected indices:**
   ```python
   import pandas as pd
   indices = pd.read_csv(
       f"{OGAL_OUTPUT}/full_exp_jan/ALIPY_RANDOM/Iris/selected_indices.csv.xz",
       compression='xz'
   )
   ```

2. **Compute selection overlap:**
   - Jaccard similarity between strategies
   - Clustering of strategies by selection patterns

3. **Correlate with dataset properties:**
   - When do strategies diverge most?
   - Does divergence predict performance differences?

### Key Files

| File | Content |
|------|---------|
| `<STRATEGY>/<DATASET>/selected_indices.csv.xz` | Queried sample indices per iteration |
| `_TS/selected_indices.parquet` | Aggregated selection data |

### Related Eva Script

```bash
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan
```

---

## Research Idea 4: Dataset Difficulty Taxonomy

**Question:** Can we cluster datasets by AL behavior and identify what makes some datasets "hard" for AL?

### Background

OGAL includes 92 datasets. Clustering them by AL behavior could reveal structural properties that affect AL performance.

### Approach

1. **Create behavior fingerprints:**
   - For each dataset: vector of strategy ranks
   - Or: average learning curve shape

2. **Cluster datasets:**
   ```python
   from sklearn.cluster import KMeans
   
   # Leaderboard: rows=datasets, cols=strategies
   lb = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/plots/final_leaderboard/rank_*.parquet")
   
   clusters = KMeans(n_clusters=5).fit_predict(lb)
   ```

3. **Characterize clusters:**
   - What do datasets in each cluster have in common?
   - Do clusters correlate with dataset metadata (size, dimensionality, imbalance)?

### Key Files

| File | Content |
|------|---------|
| `plots/final_leaderboard/*.parquet` | Strategy rankings per dataset |
| `_/<CATEGORIZER>/<DATASET>.npz` | Dataset-level features |

---

## Research Idea 5: Hyperparameter Sensitivity

**Question:** Which hyperparameters matter most? Do optimal hyperparameters vary by strategy or dataset?

### Background

OGAL varies 6 hyperparameters across experiments. The paper shows some don't matter much—but are there interaction effects?

### Approach

1. **Analyze variance decomposition:**
   - How much variance is explained by each hyperparameter?
   - Are there significant interactions?

2. **Build hyperparameter prediction model:**
   - Given dataset features, predict optimal batch size
   - Compare to fixed batch size baseline

3. **Study sensitivity by strategy:**
   - Do some strategies need careful tuning while others are robust?

### Related Eva Scripts

```bash
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan
```

---

## Research Idea 6: Failure Mode Discovery

**Question:** When and why do AL strategies fail? Can failures be predicted?

### Background

Not all experiments succeed. OGAL tracks failures in `05_failed_workloads.csv`.

### Approach

1. **Analyze failure patterns:**
   ```python
   failed = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_failed_workloads.csv")
   print(failed.groupby('EXP_STRATEGY').size().sort_values(ascending=False))
   ```

2. **Correlate with dataset properties:**
   - Do certain datasets cause more failures?
   - Are failures strategy-specific?

3. **Build failure predictor:**
   - Can we predict which experiments will fail before running?

### Key Files

| File | Content |
|------|---------|
| `05_failed_workloads.csv` | Failed experiments with error type |
| `05_started_oom_workloads.csv` | OOM-killed experiments |

---

## Research Idea 7: Budget Allocation Strategies

**Question:** Given a fixed labeling budget, how should it be allocated across AL iterations?

### Background

OGAL tests batch sizes 1–100. But what if batch size could vary during the AL process?

### Approach

1. **Simulate variable batch strategies:**
   - Large batches early, small batches later (or vice versa)
   - Use per-cycle data to reconstruct alternative trajectories

2. **Analyze batch size effects:**
   - Is batch=1 always best, or do larger batches help early on?
   - Does optimal batch size depend on dataset properties?

---

## Getting Started with Any Idea

### Standard Setup

```bash
# 1. Get the data
wget <URL_FROM_DOI>
unzip full_exp_jan.zip -d /path/to/results/

# 2. Setup environment
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# 3. Generate base artifacts
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
```

### Key Data Loading Patterns

```python
import pandas as pd
import numpy as np
import os

OGAL_OUTPUT = os.environ.get("OGAL_OUTPUT", "/path/to/results")

# Load completed experiments
done = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_done_workload.csv")

# Load time series
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")

# Load leaderboard
lb = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet")

# Load per-cycle metrics
accuracy = pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv.xz", compression='xz')
```

---

## Publishing Your Research

If you use OGAL in your research:

1. **Cite the paper:**
   ```bibtex
   @article{gonsior2025ogal,
     title={{Olympic Games of Active Learning}},
     author={Gonsior, Julius and others},
     journal={arXiv preprint arXiv:2506.03817},
     year={2025}
   }
   ```

2. **Cite the dataset:**
   ```
   DOI: 10.25532/OPARA-862
   ```

3. **Share your code:** Help others reproduce and extend your work

---

## Next Steps

| Goal | Page |
|------|------|
| Load and analyze the data | [Analyze the Dataset](analyze_dataset.md) |
| Understand the metrics | [Correlations: Paper ↔ Code](../reference/correlations_paper_to_code.md) |
| Add your own experiments | [Extend the Benchmark](extend_benchmark.md) |
