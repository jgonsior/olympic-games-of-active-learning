# Choose Your Path

**OGAL** = 4.6M Active Learning experiments archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

Pick the question that matches your goal:

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **"I want to analyze the published dataset for my own research"**

    Mine the 4.6M pre-computed results—no experiments needed.

    [:octicons-arrow-right-24: Analyze OPARA](personas/analyze_dataset.md)

-   :material-file-document-check:{ .lg .middle } **"I want to understand how the paper's results were computed"**

    Run the exact scripts that produce the paper's figures and tables.

    [:octicons-arrow-right-24: Reproduce the Paper](personas/reproduce_paper.md)

-   :material-plus-box:{ .lg .middle } **"I want to extend the dataset with new strategies/hyperparameters"**

    Add your experiments and integrate them with the shared benchmark.

    [:octicons-arrow-right-24: Extend the Benchmark](personas/extend_benchmark.md)

-   :material-server:{ .lg .middle } **"I want to recompute the entire dataset from scratch"**

    Run millions of experiments on HPC/SLURM, handle failures, resume.

    [:octicons-arrow-right-24: Run from Scratch](personas/run_from_scratch.md)

-   :material-code-tags:{ .lg .middle } **"I want deep codebase understanding"**

    Architecture, data model, design rationale—all in one place.

    [:octicons-arrow-right-24: Architecture & Design](personas/understand_codebase.md)

-   :material-lightbulb:{ .lg .middle } **"I want research ideas"**

    Open questions and unexplored directions using OGAL data.

    [:octicons-arrow-right-24: Research Ideas](personas/research_ideas.md)

</div>

---

## 5-Minute Win

```bash
# Setup + leaderboard from archived data
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

---

## Links

📄 [Paper](https://arxiv.org/abs/2506.03817) ・ 📦 [Dataset (DOI)](https://doi.org/10.25532/OPARA-862) ・ 💻 [GitHub](https://github.com/jgonsior/olympic-games-of-active-learning)
