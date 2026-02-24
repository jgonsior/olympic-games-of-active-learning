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

Run any script with:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

All scripts read `OUTPUT_PATH` from `.server_access_credentials.cfg` automatically.
