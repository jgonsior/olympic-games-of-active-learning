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

## Quickstart

Most users should **download the pre-computed OPARA results** rather than
re-running experiments. The documentation home page walks you through
downloading, extracting, and verifying the archive in about 10 minutes:

> **📦 [Start here](https://jgonsior.github.io/olympic-games-of-active-learning/)** — download OPARA results and begin analysing.

## Links

- 📖 [**Full documentation**](https://jgonsior.github.io/olympic-games-of-active-learning/)
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
