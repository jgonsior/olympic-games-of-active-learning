# OGAL – Survey of Active Learning Hyperparameters

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
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# edit .server_access_credentials.cfg → set OUTPUT_PATH and DATASETS_PATH under [LOCAL]

# 2. Analyze pre-computed results (no experiments needed)
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
unzip full_exp_jan.zip -d /path/to/results/
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan

# 3. Or run your own experiment
python 01_create_workload.py --EXP_TITLE test
# This creates several files in OUTPUT_PATH/test/, including:
#   01_workload.csv              – hyperparameter grid (one row per experiment)
#   02b_run_bash_parallel.py     – script to run all experiments in parallel locally
# Run all experiments locally in parallel:
python "$OUTPUT_PATH/test/02b_run_bash_parallel.py"
```

## Links

- 📖 [**Documentation**](https://jgonsior.github.io/olympic-games-of-active-learning/) — Start here
- 📊 [**Analyze the dataset**](https://jgonsior.github.io/olympic-games-of-active-learning/personas/analyze_dataset/) — Research tutorials
- 📄 [**Paper (arXiv:2506.03817)**](https://arxiv.org/abs/2506.03817) — Methodology and findings
- 📦 [**Archived data (DOI)**](https://doi.org/10.25532/OPARA-862) — 4.6M experiment results

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.
For the released dataset, cite [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).

```bibtex
@misc{gonsior2025surveyactivelearninghyperparameters,
  title={{Survey of Active Learning Hyperparameters: Insights from a Large-Scale Experimental Grid}},
  author={Julius Gonsior and Tim Rie{\ss} and Anja Reusch and Claudio Hartmann and Maik Thiele and Wolfgang Lehner},
  year={2025},
  eprint={2506.03817},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.03817}
}

@dataset{gonsior2025ogal_dataset,
  author={Julius Gonsior and Tim Rie{\ss} and Anja Reusch and Claudio Hartmann and Maik Thiele and Wolfgang Lehner},
  title={{OGAL: Survey of Active Learning Hyperparameters -- Dataset}},
  year={2025},
  publisher={OPARA},
  doi={10.25532/OPARA-862},
  url={https://doi.org/10.25532/OPARA-862}
}
```

## License

[AGPL-3.0](LICENSE). The `LICENSE` file is authoritative; packaging metadata may be inconsistent.
