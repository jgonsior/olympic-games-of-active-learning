# Start Here

**OGAL** = largest Active Learning benchmark: **4.6M experiments** archived and ready to use.

!!! success "Skip 3.6 million CPU hours"
    Download results from **[DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)** and analyze without rerunning.

---

## What Do You Want to Do?

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } **Analyze the Dataset**

    Use 4.6M pre-computed results for your research.

    [:octicons-arrow-right-24: Analyze the dataset](analyze_dataset.md)

-   :material-plus-circle:{ .lg .middle } **Add Your Results**

    Contribute new experiments to the benchmark.

    [:octicons-arrow-right-24: Add your results](add_results.md)

-   :material-cog:{ .lg .middle } **Run Experiments**

    Execute locally or on HPC/SLURM.

    [:octicons-arrow-right-24: Runbook](reference/runbook.md)

</div>

---

## Minimal Example

```bash
# Setup + generate leaderboard from archived data (no experiments needed)
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

See [Analyze the Dataset](analyze_dataset.md) for research tutorials.
