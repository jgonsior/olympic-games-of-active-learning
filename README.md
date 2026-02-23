# OGAL – Survey of Active Learning Hyperparameters

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://jgonsior.github.io/olympic-games-of-active-learning/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03817-b31b1b.svg)](https://arxiv.org/abs/2506.03817)
[![DOI](https://img.shields.io/badge/DOI-10.25532%2FOPARA--862-blue)](https://doi.org/10.25532/OPARA-862)

## Why OGAL?

- **4.6M pre-computed experiments** — skip ~3.6 million CPU hours of compute
- **Unified API** across multiple AL frameworks (ALiPy, libact, small-text, scikit-activeml, Playground) plus OGAL-native baselines
- **Consistent protocol** — same splits, seeds, and output schema for all strategies
- **Reusable dataset** archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)
- **Ready-to-use analysis scripts** (`eva_scripts/`) for leaderboards, correlations, and paper figures

## Quickstart

```bash
# 1. Install
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install

# 2. Configure local paths (required for all local runs)
#    Copy the committed template and fill in your absolute paths first.
#    The filled-in file is gitignored — never commit it.
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# Edit .server_access_credentials.cfg:
#   Under [LOCAL], set OUTPUT_PATH and DATASETS_PATH to real absolute paths.

# 3. Download the released OPARA results (~320 GB) — full instructions:
#    https://jgonsior.github.io/olympic-games-of-active-learning/use_opara_archive/
#    Quick version (wget, resumable):
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
unzip full_exp_jan.zip -d "${RESULTS_DIR}/full_exp_jan"  # RESULTS_DIR = your OUTPUT_PATH

# 4. Analyze pre-computed results
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan

# 5. Or run your own experiment
python 01_create_workload.py --EXP_TITLE test && python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

> **📦 Full OPARA download guide** (both artifacts, aria2c alternative, verification):
> [Use released OPARA archive](https://jgonsior.github.io/olympic-games-of-active-learning/use_opara_archive/)

## Reproducing the paper run (`full_exp_jan`)

> **Compute scale:** The paper run covers ~4.6 M experiments (~3.6 M CPU hours on HPC).
> Running it on a laptop is not feasible.  For a local sanity check, use the `test`
> config (completes in minutes).  For the full experiment grid, strategy list, and
> evaluation scripts, see [Paper subset](https://jgonsior.github.io/olympic-games-of-active-learning/paper_subset/).

The paper results come from config **`full_exp_jan`** in `resources/exp_config.yaml`.
**The archived results are already on OPARA** (downloaded above) — re-running from scratch
requires an HPC cluster.  See [Pipeline](https://jgonsior.github.io/olympic-games-of-active-learning/pipeline/)
for the full step-by-step commands.

## Links
- 📊 [**Analyze the dataset**](https://jgonsior.github.io/olympic-games-of-active-learning/personas/analyze_dataset/) — Research tutorials
- 📄 [**Paper (arXiv:2506.03817)**](https://arxiv.org/abs/2506.03817) — Methodology and findings
- 📦 [**Archived data (DOI)**](https://doi.org/10.25532/OPARA-862) — 4.6M experiment results

## Citation

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.
For the released dataset, cite [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862).
To cite a specific version, include the git commit hash or tag you used.

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
