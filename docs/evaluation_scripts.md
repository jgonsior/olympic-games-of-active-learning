# Evaluation Scripts

Scripts in the `eva_scripts/` directory analyze experiment results and produce the paper's figures and tables.

---

## Key scripts

| Script | What it does |
|--------|-------------|
| `eva_scripts.final_leaderboard` | Strategy ranking table (Table 1 in the paper) |
| `eva_scripts.calculate_dataset_dependend_random_ramp_slope` | Random baseline slope for normalised rankings |
| `eva_scripts.workload_reduction` | Pearson-$r$ heatmaps (metric correlation) |
| `eva_scripts.single_hyperparameter_evaluation_indices` | Jaccard-$J$ heatmaps (sample selection overlap) |
| `eva_scripts.leaderboard_single_hyperparameter_influence` | Kendall-$\tau_b$ heatmaps (ranking stability) |

## Running an evaluation script

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

All scripts read `OUTPUT_PATH` from `.server_access_credentials.cfg` automatically.

---

**See also:** [Results format](results_format.md) · [Pipeline](pipeline.md#step-6-generate-leaderboard) · [Paper subset](paper_subset.md)
