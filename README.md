# OGAL – Olympic Games of Active Learning

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://jgonsior.github.io/olympic-games-of-active-learning/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.03817-b31b1b.svg)](https://arxiv.org/abs/2506.03817)
[![DOI](https://img.shields.io/badge/DOI-10.25532%2FOPARA--862-blue)](https://doi.org/10.25532/OPARA-862)

> **Note**: This documentation was generated with the assistance of GitHub Copilot (Claude AI) based on analysis of the codebase, research paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)), and archived data ([DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)). While efforts were made to ensure accuracy, users should verify critical details against the source code and paper.

---

## Value of the published dataset

### Why OGAL?
- Unified multi-framework AL benchmarking runner with a single API and consistent protocol for 50+ strategies (see [`resources/exp_config.yaml`](resources/exp_config.yaml)).
- Shared archived experimental dataset [OPARA-862](https://doi.org/10.25532/OPARA-862) that can be reused and extended via the documented enrichment protocol.
- [`eva_scripts/`](docs/eva_scripts.md) as the analysis API to turn raw runs into derived artifacts and plots.
- **Compute saved:** Reproducing the full hyperparameter grid from scratch costs **~3.6 million CPU hours** (paper: [arXiv:2506.03817](https://arxiv.org/abs/2506.03817)).

### What's inside the dataset?
- Preview: [Dataset value](docs/dataset_value.md) · [Data enrichment](docs/data_enrichment.md) · [Eva scripts](docs/eva_scripts.md)

### Research opportunities
- See [Research opportunities](docs/research_opportunities.md) for concrete analyses enabled by the archived data and eva scripts.

---

## Unified Active Learning Benchmarking API

OGAL's key innovation is providing a **single experiment API** that runs strategies from multiple AL frameworks under a **consistent protocol**:

- **Same train/test splits** across all strategies and datasets
- **Same random seeds** for reproducibility
- **Same budget/batch configurations**
- **Same output schema** (metrics, selected indices, timing)

```python
# Configure by strategy NAME, not framework details
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_STRATEGY:
    - ALIPY_UNCERTAINTY_LC      # ALiPy framework
    - LIBACT_QUIRE              # libact framework
    - SMALLTEXT_EMBEDDINGKMEANS # small-text framework
    - SKACTIVEML_QBC            # scikit-activeml framework
  # OGAL handles framework dispatch internally
```

See the **[Strategy Catalog](https://jgonsior.github.io/olympic-games-of-active-learning/strategies/)** for all 50+ available strategies.

---

## Core Concepts

Understand OGAL's internal architecture and design:

| Concept | Description |
|---------|-------------|
| **[Architecture](https://jgonsior.github.io/olympic-games-of-active-learning/architecture/)** | System design, core abstractions, failure recovery, HPC parallelism |
| **[Configuration](https://jgonsior.github.io/olympic-games-of-active-learning/configuration/)** | Config files, data flow, environment setup |
| **[Data Model](https://jgonsior.github.io/olympic-games-of-active-learning/data_model/)** | Run identity, workload schema, result schemas, enum definitions |
| **[Dataset Metadata](https://jgonsior.github.io/olympic-games-of-active-learning/dataset_metadata/)** | Auto-computed per-sample categorizations |
| **[Utilities](https://jgonsior.github.io/olympic-games-of-active-learning/utilities/)** | Helper modules and scripts catalog |

---

## Key Documentation Links

| Document | Purpose |
|----------|---------|
| 📋 **[Strategy Catalog](https://jgonsior.github.io/olympic-games-of-active-learning/strategies/)** | Complete list of all AL strategies (strategy-first reference) |
| 📈 **[Eva Scripts](https://jgonsior.github.io/olympic-games-of-active-learning/eva_scripts/)** | Analyze experiment results and generate paper figures |
| ➕ **[Data Enrichment](https://jgonsior.github.io/olympic-games-of-active-learning/data_enrichment/)** | Protocol for adding new results to the shared dataset |
| 🖥️ **[HPC Runbook](https://jgonsior.github.io/olympic-games-of-active-learning/hpc/)** | Running large-scale experiments on SLURM clusters |
| 📦 **[DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)** | Archived experiment results (terabytes of raw data) |

---

## Overview

This repository contains the complete experimental framework used for the **largest conducted AL study to date** with over **4.6 million hyperparameter combinations**. The pipeline evaluates:

- **50+ AL strategies** from 5 frameworks (ALiPy, libact, small-text, scikit-activeml, playground)
- **92 datasets** from OpenML, Kaggle, and UCI (100-20,000 samples, 2-31 classes)
- **3 learner models**: Random Forest, MLP, SVM with RBF kernel
- **6 batch sizes**: 1, 5, 10, 20, 50, 100
- **5 train-test splits × 20 start points** per dataset (100 repetitions)

The study analyzes the impact of each hyperparameter on AL experiment results and provides recommendations for reproducible AL research.

---

## Paper & Archived Artifacts

| Resource | Description |
|----------|-------------|
| **Research Paper** | [arXiv:2506.03817](https://arxiv.org/abs/2506.03817) - Methodology, experimental design, and findings |
| **GitHub Repository** | [jgonsior/olympic-games-of-active-learning](https://github.com/jgonsior/olympic-games-of-active-learning) - Source code and experiment pipeline |
| **Archived Data (DOI)** | [10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862) - Raw experiment results (~terabytes) |

**What's in the DOI archive vs this repo?**

- **OPARA-862 (DOI)**: Raw experiment outputs (metrics CSVs, selected indices, timing data) - the shared dataset
- **This repo**: Code to run experiments, post-processing scripts, eva_scripts for analysis

---

## Supported Strategies (Preview)

| Strategy | Family | Framework |
|----------|--------|-----------|
| `ALIPY_UNCERTAINTY_LC` | Uncertainty | ALiPy |
| `ALIPY_UNCERTAINTY_ENTROPY` | Uncertainty | ALiPy |
| `LIBACT_QUIRE` | Query-specific | libact |
| `LIBACT_DWUS` | Density-weighted | libact |
| `SMALLTEXT_EMBEDDINGKMEANS` | Diversity | small-text |
| `SKACTIVEML_QBC` | Committee | scikit-activeml |
| `SKACTIVEML_MC_EER_LOG_LOSS` | Expected Error | scikit-activeml |
| `PLAYGROUND_KCENTER_GREEDY` | Diversity | Playground |

➡️ **[See all 50+ strategies in the Strategy Catalog](https://jgonsior.github.io/olympic-games-of-active-learning/strategies/)**

---

## Sequential Pipeline

OGAL is a **strictly sequential pipeline**. Each script produces outputs consumed by the next step. The shared configuration (`resources/exp_config.yaml` and `.server_access_credentials.cfg`) is the source of truth for data flow.

### Run Order

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `00_download_datasets.py` | Download/prepare datasets from OpenML and Kaggle |
| 1 | `01_create_workload.py` | Generate experiment workload grid (datasets × strategies × models × seeds) |
| 2 | `02_run_experiment.py` | Execute AL experiments (supports worker sharding for HPC) |
| 3 | `03_calculate_dataset_categorizations.py` | Compute per-sample categorizations (hardness metrics) |
| 4 | `04_calculate_advanced_metrics.py` | Compute derived metrics (AUC, time-lag, distance metrics) |
| 5 | `05_analyze_partially_run_workload.py` | Analyze completion status and timing statistics |
| 6 | `07b_create_results_without_flask.py` | Generate final visualizations and result tables |

Additionally, `scripts/` and `eva_scripts/` provide utility and evaluation scripts for data processing and paper figure generation.

For detailed documentation on each step, see [docs/pipeline.md](docs/pipeline.md).

## Quickstart: Local Sanity Run

A minimal local experiment to verify installation:

```bash
# 1. Setup environment
conda create --name al_olympics_env --file conda-linux-64.lock
conda activate al_olympics_env
poetry install

# 2. Configure paths (create .server_access_credentials.cfg)
# See "Configuration" section below

# 3. Download datasets (or use existing)
python 00_download_datasets.py

# 4. Create a small workload using the test config
python 01_create_workload.py --EXP_TITLE test

# 5. Run a single experiment (WORKER_INDEX selects row from workload)
python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

## Compute Reality Check

| Environment | Recommended For |
|-------------|-----------------|
| **Local** | Development, debugging, small experiments (1-100 runs) |
| **HPC/SLURM** | Full experiments (thousands to millions of runs) |

A full paper-scale experiment may involve **millions of individual AL runs** across the hyperparameter grid. This is designed for HPC clusters with SLURM job arrays.

## Installation

### Requirements

- [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or Miniconda
- Python 3.11

### Local Installation

```bash
# In the repository root
conda create --name al_olympics_env --file conda-linux-64.lock
conda activate al_olympics_env
poetry install
```

### HPC Cluster Installation

```bash
# Replace $WS_URL with your workspace path
module load release/25.06
module load Anaconda3/2025.06-1
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda create --prefix $WS_URL/conda-env --file conda-linux-64.lock
conda activate $WS_URL/conda-env
poetry install
```

## Configuration

### `.server_access_credentials.cfg`

Create this file in the repository root (it's gitignored for security):

```ini
[HPC]
SSH_LOGIN=user@hpc_server
WS_PATH=/path/to/workspace
DATASETS_PATH=/path/to/datasets
OUTPUT_PATH=/path/to/results
SLURM_MAIL=your.email@example.org
SLURM_PROJECT="your_project_name"
CODE_PATH=/path/to/code
PYTHON_PATH=/path/to/python

[LOCAL]
DATASETS_PATH=/home/user/al_survey/datasets
CODE_PATH=/home/user/al_survey/code
OUTPUT_PATH=/home/user/al_survey/exp_results
```

### Experiment Configuration (`resources/exp_config.yaml`)

Define experiment grids using YAML:

```yaml
test:
  EXP_GRID_DATASET: [Iris, wine_origin]
  EXP_GRID_STRATEGY: [ALIPY_RANDOM, ALIPY_UNCERTAINTY_LC]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_NUM_QUERIES: [10]
  EXP_GRID_BATCH_SIZE: [1, 5]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
  EXP_GRID_START_POINT: [0-4]
  METRICS: [Standard_ML_Metrics, Selected_Indices, Timing_Metrics]
```

## Repository Map

### Active Directories

| Directory | Purpose |
|-----------|---------|
| `datasets/` | Dataset loaders (OpenML, Kaggle, local) |
| `framework_runners/` | AL framework adapters (ALiPy, libact, small-text, scikit-activeml) |
| `optimal_query_strategies/` | Oracle/optimal strategy implementations |
| `metrics/` | Metric computation (per-cycle and post-hoc) |
| `resources/` | Configuration files and SLURM templates |
| `scripts/` | Utility scripts for data processing and fixes |
| `eva_scripts/` | Evaluation and visualization scripts |
| `misc/` | Shared utilities (config, logging, helpers) |

### Deprecated Directories

| Directory | Status |
|-----------|--------|
| `analyse_results/` | **Deprecated / not used.** Legacy visualization code. |

## Troubleshooting

### Common Issues

1. **Missing `.server_access_credentials.cfg`**
   - Create the file following the template above
   - Ensure paths exist and are writable

2. **Dataset download failures**
   - Check Kaggle API credentials (~/.kaggle/kaggle.json)
   - Verify OpenML API connectivity

3. **Out-of-memory (OOM) errors**
   - Reduce batch size or dataset size
   - Check `05_started_oom_workloads.csv` for tracked OOM runs

4. **Experiment not resuming**
   - The framework tracks done/failed workloads automatically
   - Re-running `01_create_workload.py` will exclude completed experiments

### How to Resume Partial Runs

```bash
# Re-run workload creation to identify remaining work
python 01_create_workload.py --EXP_TITLE your_experiment

# Continue with the next available WORKER_INDEX
python 02_run_experiment.py --EXP_TITLE your_experiment --WORKER_INDEX <next_index>
```

The framework maintains:
- `05_done_workload.csv`: Completed experiments
- `05_failed_workloads.csv`: Failed experiments (with error types)
- `05_started_oom_workloads.csv`: OOM-killed experiments

## Citation

If you use OGAL in your research, please cite both the paper and the repository:

```bibtex
@article{gonsior2025ogal,
  title={{Olympic Games of Active Learning: A Large-Scale Empirical Study of Active Learning Strategies}},
  author={Gonsior, Julius and Rie{\ss}, Tim and Reusch, Anja and Hartmann, Claudio and Thiele, Maik and Lehner, Wolfgang},
  journal={arXiv preprint arXiv:2506.03817},
  year={2025},
  url={https://arxiv.org/abs/2506.03817}
}

@software{ogal2025code,
  author = {Gonsior, Julius and Rie{\ss}, Tim and Reusch, Anja and Hartmann, Claudio and Thiele, Maik and Lehner, Wolfgang},
  title = {{OGAL}: Olympic Games of Active Learning - Code Repository},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jgonsior/olympic-games-of-active-learning},
  doi = {10.25532/OPARA-862}
}
```

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for details.

## Contributing

See [docs/contributing.md](docs/contributing.md) for development setup and contribution guidelines.
