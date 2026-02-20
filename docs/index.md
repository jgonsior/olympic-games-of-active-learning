# Choose Your Path

**OGAL** = 4.6M Active Learning experiments archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

Pick the question that matches your goal:

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **"I want to analyze the published dataset for my own research"**

    Mine the 4.6M pre-computed results—no experiments needed.

    [:octicons-arrow-right-24: Analyze OPARA](personas/analyze_dataset.md)

-   :material-file-document-check:{ .lg .middle } **"I want to reproduce the paper or run experiments from scratch"**

    Run the exact scripts for the paper's figures, or recompute on HPC/SLURM.

    [:octicons-arrow-right-24: Reproduce & Run](personas/reproduce_and_run.md)

-   :material-plus-box:{ .lg .middle } **"I want to extend the dataset with new strategies/hyperparameters"**

    Add your experiments and integrate them with the shared benchmark.

    [:octicons-arrow-right-24: Extend the Benchmark](personas/extend_benchmark.md)

-   :material-lightbulb:{ .lg .middle } **"I want research ideas"**

    Open questions and unexplored directions using OGAL data.

    [:octicons-arrow-right-24: Research Ideas](personas/research_ideas.md)

</div>

---

## Getting Started

```bash
# Setup + leaderboard from archived data
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

---

## Links

📄 [Paper](https://arxiv.org/abs/2506.03817) ・ 📦 [Dataset (DOI)](https://doi.org/10.25532/OPARA-862) ・ 💻 [GitHub](https://github.com/jgonsior/olympic-games-of-active-learning)
