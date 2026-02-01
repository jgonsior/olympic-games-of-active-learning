# OGAL – Olympic Games of Active Learning

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://jgonsior.github.io/olympic-games-of-active-learning/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03817-b31b1b.svg)](https://arxiv.org/abs/2506.03817)
[![DOI](https://img.shields.io/badge/DOI-10.25532%2FOPARA--862-blue)](https://doi.org/10.25532/OPARA-862)

## Why OGAL?

- **4.6M pre-computed experiments** — skip ~3.6 million CPU hours of compute
- **Unified API** for 50+ AL strategies across 5 frameworks (ALiPy, libact, small-text, scikit-activeml, playground)
- **Consistent protocol** — same splits, seeds, and output schema for all strategies
- **Reusable dataset** archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)
- **Ready-to-use analysis scripts** (`eva_scripts/`) for leaderboards, correlations, and paper figures

## Quickstart

```bash
# 1. Install
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# 2. Analyze pre-computed results (no experiments needed)
wget <URL_FROM_DOI> && unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan

# 3. Or run your own experiment
python 01_create_workload.py --EXP_TITLE test && python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

## Links

- 📖 [**Documentation**](https://jgonsior.github.io/olympic-games-of-active-learning/) — Start here
- 📊 [**Analyze the dataset**](https://jgonsior.github.io/olympic-games-of-active-learning/analyze_dataset/) — Research tutorials
- 📄 [**Paper (arXiv:2506.03817)**](https://arxiv.org/abs/2506.03817) — Methodology and findings
- 📦 [**Archived data (DOI)**](https://doi.org/10.25532/OPARA-862) — 4.6M experiment results
- 🤝 [**Contributing**](https://jgonsior.github.io/olympic-games-of-active-learning/contributing/) — Development guide

## Citation

```bibtex
@misc{gonsior2025ogal,
  title={{Olympic Games of Active Learning: A Large-Scale Empirical Study of Active Learning Strategies}},
  author={Gonsior, Julius and Rie{\ss}, Tim and Reusch, Anja and Hartmann, Claudio and Thiele, Maik and Lehner, Wolfgang},
  year={2025},
  eprint={2506.03817},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@dataset{gonsior2025ogal_dataset,
  author={Gonsior, Julius and Rie{\ss}, Tim and Reusch, Anja and Hartmann, Claudio and Thiele, Maik and Lehner, Wolfgang},
  title={{OGAL: Olympic Games of Active Learning -- Dataset}},
  year={2025},
  publisher={OPARA},
  doi={10.25532/OPARA-862},
  url={https://doi.org/10.25532/OPARA-862}
}
```

## License

[AGPL-3.0](LICENSE)
