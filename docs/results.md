# Results & evaluation

How OGAL stores experiment results and how to analyze them.

---

## File lifecycle

| Phase | Format | Why |
|-------|--------|-----|
| **During runs** | Append-only `.csv` | Thousands of parallel HPC workers safely append rows to shared files |
| **After runs / for release** | Compressed `.csv.xz` | ~10× smaller; faster to distribute and evaluate |
| **Evaluation scripts** | Read `.csv.xz` if present, fall back to `.csv` | Works with both in-progress and archived results |

---

## Schema

Metric files live under `{OUTPUT_PATH}/{EXP_TITLE}/{STRATEGY}/{DATASET}/`.

Each file (e.g. `accuracy.csv.xz`) has one **row per experiment** with:

| Column | Description |
|--------|-------------|
| `EXP_UNIQUE_ID` | Links back to the workload CSVs (`01_workload.csv`, `05_done_workload.csv`) |
| `0`, `1`, … `N-1` | Metric value at each AL cycle |

Workload tracking files (`01_workload.csv`, `05_done_workload.csv`, `05_failed_workloads.csv`) are plain CSVs in `{OUTPUT_PATH}/{EXP_TITLE}/` — see [Reference](reference.md) for their columns.

---

## Reading results in Python

**Read a compressed `.csv.xz`:**

```python
import pandas as pd

acc = pd.read_csv(
    "results/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv.xz",
    compression="xz",
)
```

**Read a raw `.csv` (mid-run or before compression):**

```python
acc = pd.read_csv(
    "results/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv",
)
```

**Helper — prefer `.csv.xz`, fall back to `.csv`:**

```python
from pathlib import Path

def read_metric(path: str) -> pd.DataFrame:
    p = Path(path)
    xz = p.with_suffix(p.suffix + ".xz")
    if xz.exists():
        return pd.read_csv(xz, compression="xz")
    return pd.read_csv(p)
```

---

## Evaluation scripts

Scripts in `eva_scripts/` consume result files and produce the paper's figures and tables.

| Script | What it produces |
|--------|-----------------|
| `eva_scripts.final_leaderboard` | Strategy ranking table (Table 1) |
| `eva_scripts.calculate_dataset_dependend_random_ramp_slope` | Random baseline slope for normalised rankings |
| `eva_scripts.workload_reduction` | Pearson-*r* heatmaps (metric correlation) |
| `eva_scripts.single_hyperparameter_evaluation_indices` | Jaccard-*J* heatmaps (sample selection overlap) |
| `eva_scripts.leaderboard_single_hyperparameter_influence` | Kendall-*τ*<sub>b</sub> heatmaps (ranking stability) |

Run any script with:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

All scripts read `OUTPUT_PATH` from `.server_access_credentials.cfg` automatically.

---

## Paper → code: correlation metrics

The paper (§IV-B) uses three correlation methods to quantify hyperparameter influence.
Each method has a **colour-coded heatmap** and a corresponding evaluation script.

| Paper section | Correlation | Heatmap colour | Script | What it measures |
|---------------|-------------|----------------|--------|-----------------|
| §IV-B1 | Pearson *r* | **Blue** | `eva_scripts.workload_reduction` | Do two hyperparameter values produce similar ML metric outcomes? |
| §IV-B2 | Jaccard *J* | **Green** | `eva_scripts.single_hyperparameter_evaluation_indices` | Do strategies select the same samples for labelling? |
| §IV-B3 | Kendall *τ*<sub>b</sub> | **Orange** | `eva_scripts.leaderboard_single_hyperparameter_influence` | Do strategy rankings stay the same when a hyperparameter changes? |

### Metric-based (Pearson *r*, §IV-B1)

For each value of a hyperparameter (e.g. batch size $b_i$), a result vector
$V_{b_i}(M)$ is built from all experiments sharing that value.
Pairwise Pearson *r* between these vectors fills the blue heatmap.
High *r* ≈ changing the hyperparameter has little effect on the metric.

$$
V_{b_i}(M) = \begin{bmatrix} M_{b_i 1} \\ M_{b_i 2} \\ \vdots \end{bmatrix}
\qquad
\text{Heatmap cell} = r\!\bigl(V_{b_i}(M),\; V_{b_j}(M)\bigr)
$$

### Queried samples (Jaccard *J*, §IV-B2)

Each experiment's per-cycle queried sets are unioned into $\widehat{Q}$.
Pairwise Jaccard similarity is averaged across matched experiment pairs.
High similarity means the same samples were selected for labelling.
See the paper (§IV-B2) for the full derivation of the heatmap values.

$$
\widehat{Q} = \bigcup_{i=0}^{c} Q^i
\qquad
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

### Ranking invariance (Kendall *τ*<sub>b</sub>, §IV-B3)

A leaderboard (strategies × datasets) is built, averaged to a ranking vector
per hyperparameter value, then compared with Kendall *τ*<sub>b</sub>.
The orange heatmap shows whether strategy rankings are stable under
hyperparameter changes.

$$
\tau_b = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}
$$

where $n_c$ = concordant pairs, $n_d$ = discordant pairs.

### Data flow

All three methods read `_TS/{metric}.parquet` files (auto-generated on first
evaluation run).  These join per-cycle `.csv.xz` metrics with workload metadata
from `05_done_workload.csv`.
