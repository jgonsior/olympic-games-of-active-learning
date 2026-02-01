# Start Here

**OGAL** = largest Active Learning benchmark: **4.6M experiments** archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

---

## What Do You Want to Do?

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } **Analyze OPARA Data**

    Use 4.6M pre-computed results for your research.

    [:octicons-arrow-right-24: Analyze OPARA](analyze_opara.md)

-   :material-plus-circle:{ .lg .middle } **Add Your Results**

    Contribute new experiments to the benchmark.

    [:octicons-arrow-right-24: Add results](add_results.md)

-   :material-cog:{ .lg .middle } **Run Experiments**

    Execute locally or on HPC/SLURM.

    [:octicons-arrow-right-24: Runbook](reference/runbook.md)

</div>

---

## 5-Minute Win

```bash
# Setup + generate leaderboard from archived data
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
# → plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet
```
