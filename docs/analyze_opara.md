# Analyze OPARA

Use the published OPARA dataset (4.6M experiments) for research without rerunning anything.

---

## Setup (5 minutes)

```bash
# 1) Get archived data
wget <URL_FROM_DOI_LANDING_PAGE>
unzip full_exp_jan.zip -d /path/to/results/full_exp_jan

# 2) Environment
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# 3) Paths
export OGAL_OUTPUT=/path/to/results
cat > .server_access_credentials.cfg << 'EOF'
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF
```

---

## C. Research starter kit

### C1. Stopping Point Analysis

Goal: detect when AL should stop.

```bash
python -m eva_scripts.learning_curve --EXP_TITLE full_exp_jan
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
```

Outputs: `_TS/weighted_f1-score.parquet`, `plots/single_learning_curve/*.parquet`

---

### C2. Strategy Recommendation / Meta-Learning

Goal: recommend AL strategies from dataset features.

```bash
python << 'EOF'
import pandas as pd, os
OGAL_OUTPUT = os.environ["OGAL_OUTPUT"]
ts = pd.read_parquet(f"{OGAL_OUTPUT}/full_exp_jan/_TS/full_auc_weighted_f1-score.parquet")
print(ts.groupby(['EXP_DATASET','EXP_STRATEGY'])['metric_value'].mean().head())
EOF
```

Next: train a meta-model; use early AUC (`first_5_*.csv.xz`) as cheap proxy.

---

### C3. Dataset similarity clustering

Goal: cluster datasets by AL behavior.

```bash
python << 'EOF'
from pathlib import Path; import os
OGAL_OUTPUT = os.environ["OGAL_OUTPUT"]
print(len(list(Path(f"{OGAL_OUTPUT}/full_exp_jan/_/REGION_DENSITY/").glob('*.npz'))))
EOF
```

Next: pair with categorizations in `_/<CATEGORIZER>/`.

---

### C4. Trade-off analysis

Goal: accuracy vs F1 vs runtime.

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
```

Next: compare heatmaps per strategy.

---

### C5. Failure mode discovery

Goal: find failing strategies.

```bash
python << 'EOF'
import pandas as pd, os
OGAL_OUTPUT=os.environ["OGAL_OUTPUT"]
failed=pd.read_csv(f"{OGAL_OUTPUT}/full_exp_jan/05_failed_workloads.csv")
print(failed.groupby('EXP_STRATEGY').size().sort_values(ascending=False).head())
EOF
```

Next: correlate with dataset size/imbalance.

---

## A. Produce key artifacts (minimal chain)

```bash
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.calculate_leaderboard_rankings --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

Output: `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet`

---

## B. Three correlations (paper)

| Correlation | Command | Next |
|-------------|---------|------|
| Metric (Pearson) | `python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan` | See [Correlations](reference/correlations_paper_to_code.md) |
| Queried samples (Jaccard) | `python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan` | 〃 |
| Ranking invariance (Kendall) | `python -m eva_scripts.leaderboard_single_hyperparameter_influence --EXP_TITLE full_exp_jan` | 〃 |

---

## Reference

- Eva scripts: [Reference](reference/eva_scripts_catalog.md)
- Results schema: [Reference](reference/results_schema.md)
- Strategies overview: [Reference](reference/strategy_overview.md)
- Run experiments: [Runbook](reference/runbook.md)
