# Start Here

**OGAL** = 4.6M archived Active Learning experiments: [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

---

## Pick your path

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } **Analyze OPARA**

    Use the archived 4.6M results.

    [:octicons-arrow-right-24: Analyze OPARA](analyze_opara.md)

-   :material-plus-circle:{ .lg .middle } **Add results**

    Contribute new experiments.

    [:octicons-arrow-right-24: Add results](add_results.md)

-   :material-cog:{ .lg .middle } **Runbook**

    Minimal guide to run OGAL.

    [:octicons-arrow-right-24: Runbook](reference/runbook.md)

</div>

---

## 5-minute win (leaderboard)

```bash
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
# → plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet
```
