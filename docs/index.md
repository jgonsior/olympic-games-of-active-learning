# Welcome to OGAL

**OGAL** (Olympic Games of Active Learning) is the largest Active Learning benchmark: **4.6M experiments** archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

---

## Who Are You?

Choose the path that best describes your goal:

<div class="grid cards" markdown>

-   :material-chart-line:{ .lg .middle } **"I want to analyze the dataset for my research"**

    Load the 4.6M pre-computed results and use them directly in your own research—no need to run experiments yourself.

    [:octicons-arrow-right-24: Analyze the Dataset](personas/analyze_dataset.md)

-   :material-file-document-check:{ .lg .middle } **"I want to reproduce the paper results"**

    Understand how the figures, tables, and conclusions in the paper were computed. Run the exact same evaluation scripts.

    [:octicons-arrow-right-24: Reproduce Paper Results](personas/reproduce_paper.md)

-   :material-code-tags:{ .lg .middle } **"I want to understand the complete codebase"**

    Dive deep into the architecture, data model, and design decisions. Understand every script and module.

    [:octicons-arrow-right-24: Understand the Codebase](personas/understand_codebase.md)

-   :material-plus-box:{ .lg .middle } **"I want to extend the benchmark"**

    Add new AL strategies, datasets, learner models, or hyperparameters. Contribute back to the shared dataset.

    [:octicons-arrow-right-24: Extend the Benchmark](personas/extend_benchmark.md)

-   :material-lightbulb:{ .lg .middle } **"I'm looking for research ideas"**

    Explore open questions, unexplored corners of the dataset, and potential research directions.

    [:octicons-arrow-right-24: Research Ideas](personas/research_ideas.md)

</div>

---

## Quick Reference

| Goal | Page |
|------|------|
| Load and analyze the 4.6M archived results | [Analyze the Dataset](personas/analyze_dataset.md) |
| Reproduce paper figures and tables | [Reproduce Paper Results](personas/reproduce_paper.md) |
| Understand architecture and design | [Understand the Codebase](personas/understand_codebase.md) |
| Add new strategies/datasets/models | [Extend the Benchmark](personas/extend_benchmark.md) |
| Find research opportunities | [Research Ideas](personas/research_ideas.md) |
| Run experiments locally or on HPC | [Runbook](reference/runbook.md) |
| Development and contribution guidelines | [Contributing](contributing.md) |

---

## 5-Minute Win

```bash
# Setup + generate leaderboard from archived data
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
# → plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet
```

---

## Links

- 📄 [**Paper (arXiv:2506.03817)**](https://arxiv.org/abs/2506.03817) — Methodology and findings
- 📦 [**Archived data (DOI:10.25532/OPARA-862)**](https://doi.org/10.25532/OPARA-862) — 4.6M experiment results
- 💻 [**GitHub Repository**](https://github.com/jgonsior/olympic-games-of-active-learning)
