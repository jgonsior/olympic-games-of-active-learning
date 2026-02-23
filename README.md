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

# 2. Configure local paths (required for all local runs)
#    Copy the committed template and fill in your absolute paths first.
#    The filled-in file is gitignored — never commit it.
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# Edit .server_access_credentials.cfg:
#   Under [LOCAL], set OUTPUT_PATH and DATASETS_PATH to real absolute paths.
#   Note the OUTPUT_PATH value — you will use the same path for RESULTS_DIR below.

# 3. Download released results from OPARA (DOI: 10.25532/OPARA-862)
#    The canonical landing page is https://doi.org/10.25532/OPARA-862
#    Use -c to resume interrupted downloads. Bitstream URLs below are current as of
#    the dataset release; if OPARA migrates, retrieve updated URLs from the DOI landing page.

#    Step 3a – file manifest (~184 MB, useful for verifying the extracted archive)
wget -c -O archive_listing.txt \
  "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"

#    Step 3b – main archive (~320 GB); aria2c is an alternative for multi-connection downloads
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
# aria2c alternative: aria2c -c -o full_exp_jan.zip \
#   "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"

#    Step 3c – unpack (ensure ~320 GB free disk space before running)
#    Set RESULTS_DIR to the same value as OUTPUT_PATH in your .server_access_credentials.cfg
export RESULTS_DIR=/absolute/path/to/results
unzip full_exp_jan.zip -d "${RESULTS_DIR}/full_exp_jan"

#    Step 3d – verify: archive_listing.txt should exist and list the extracted files
ls archive_listing.txt   # must be present
# Compare extracted tree against the manifest:
# diff <(sort archive_listing.txt) <(find "${RESULTS_DIR}/full_exp_jan" -type f | sort)

# 4. Analyze pre-computed results
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan

# 5. Or run your own experiment
python 01_create_workload.py --EXP_TITLE test && python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

## Links

- 📖 [**Documentation**](https://jgonsior.github.io/olympic-games-of-active-learning/) — Start here
- 📊 [**Analyze the dataset**](https://jgonsior.github.io/olympic-games-of-active-learning/personas/analyze_dataset/) — Research tutorials
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
