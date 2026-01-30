# OGAL Documentation

> **Note**: This documentation was generated with the assistance of GitHub Copilot (Claude AI) based on analysis of the codebase, research paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)), and archived data ([DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)). While efforts were made to ensure accuracy, users should verify critical details against the source code and paper.

Welcome to the **Olympic Games of Active Learning (OGAL)** documentation. This framework provides a comprehensive benchmarking system for evaluating Active Learning query strategies at scale.

## What is OGAL?

OGAL is a large-scale experimental framework designed to systematically evaluate Active Learning (AL) strategies. As described in the research paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)), AL is rarely used in real-world applications due to complexity and lack of trust in its effectiveness. This framework addresses these challenges by:

- Compiling a **hyperparameter grid of 4.6+ million combinations**
- Recording performance across the **largest conducted AL study to date**
- Analyzing the **impact of each hyperparameter** on experiment results

### Paper Terminology

The paper defines an AL experiment as **E = (𝒮, D, 𝒯, ℐ, M, b, c, ℒ)** — a combination of hyperparameters:

| Symbol | Term | OGAL Parameter |
|--------|------|----------------|
| 𝔻 | Dataset | `EXP_GRID_DATASET` |
| 𝕊 | AL Strategy (Query Strategy) | `EXP_GRID_STRATEGY` |
| 𝕃 | Learner Model | `EXP_GRID_LEARNER_MODEL` |
| 𝔹 | Batch Size | `EXP_GRID_BATCH_SIZE` |
| 𝕋 | Train-Test-Split | `EXP_GRID_TRAIN_TEST_BUCKET_SIZE` |
| 𝕀 | Initial Start Set | `EXP_GRID_START_POINT` |
| c | AL Cycles | `EXP_GRID_NUM_QUERIES` |

The framework evaluates:

- **28 AL strategies (𝕊)** from 5 frameworks (ALiPy, libact, small-text, scikit-activeml, playground)
- **92 datasets (𝔻)** from OpenML, Kaggle, and UCI
- **3 learner models (𝕃)**: Random Forest, MLP, SVM
- **6 batch sizes (𝔹)**: 1, 5, 10, 20, 50, 100
- **5 train-test splits (𝕋) × 20 start sets (𝕀)** per dataset

## Quick Links

| Document | Description |
|----------|-------------|
| [Pipeline](pipeline.md) | Step-by-step guide to the sequential experiment pipeline |
| [Reproducing the Paper](reproducing_paper.md) | Complete workflow to reproduce OPARA archive results |
| [Scripts & Evaluation](scripts.md) | Utility scripts and evaluation analysis scripts |
| [Configuration](configuration.md) | Shared configuration system explained |
| [Results Format](results_format.md) | Output paths, file formats, and result schemas |
| [HPC Setup](hpc.md) | Running experiments on HPC clusters with SLURM |
| [Research Reuse](research_reuse.md) | Extending the framework for your research |
| [Contributing](contributing.md) | Development setup and contribution guidelines |

## Paper & Archived Artifacts

| Resource | Link |
|----------|------|
| **Research Paper** | [arXiv:2506.03817](https://arxiv.org/abs/2506.03817) - "Survey of Active Learning Hyperparameters" |
| GitHub Repository | [jgonsior/olympic-games-of-active-learning](https://github.com/jgonsior/olympic-games-of-active-learning) |
| Archived Data (DOI) | [10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862) - Raw experiment results |
| **Reproduction Guide** | [Reproducing the Paper](reproducing_paper.md) - Complete workflow to regenerate results |

### Paper Abstract

> Annotating data is a time-consuming and costly task, but it is inherently required for supervised machine learning. Active Learning (AL) is an established method that minimizes human labeling effort by iteratively selecting the most informative unlabeled samples for expert annotation. Despite being known for decades, AL is still rarely used in real-world applications due to complexity and lack of trust in its effectiveness. We hypothesize that both reasons share the same culprit: **the large hyperparameter space of AL**. This mostly unexplored hyperparameter space often leads to misleading and irreproducible AL experiment results.

The DOI reference (OPARA-862) provides the raw experiment results (~terabytes of data) for long-term preservation and reproducibility. See [Reproducing the Paper](reproducing_paper.md) for the exact configuration (`full_exp_jan`) used to generate these results.

## Pipeline Overview

OGAL follows a strict sequential execution order (source: root directory structure):

```mermaid
flowchart TD
    A[00_download_datasets.py] --> B[01_create_workload.py]
    B --> C[02_run_experiment.py]
    C --> D[03_calculate_dataset_categorizations.py]
    D --> E[04_calculate_advanced_metrics.py]
    E --> F[05_analyze_partially_run_workload.py]
    F --> G[07b_create_results_without_flask.py]
    
    subgraph Config
        CFG1[.server_access_credentials.cfg]
        CFG2[resources/exp_config.yaml]
    end
    
    subgraph Auxiliary["Auxiliary Scripts"]
        S1[scripts/]
        S2[eva_scripts/]
    end
    
    CFG1 --> A
    CFG2 --> B
    G --> S1
    G --> S2
```

See [Pipeline Documentation](pipeline.md) for complete details on each step.

## Quickstart

```bash
# 1. Setup environment
conda create --name al_olympics_env --file conda-linux-64.lock
conda activate al_olympics_env
poetry install

# 2. Configure paths (create .server_access_credentials.cfg)

# 3. Create and run a small test workload
python 01_create_workload.py --EXP_TITLE test
python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

## Repository Structure

### Active Code

| Directory | Purpose |
|-----------|---------|
| `datasets/` | Dataset loading utilities (see `datasets/__init__.py`) |
| `framework_runners/` | AL framework adapters: ALiPy, libact, small-text, scikit-activeml, playground (see `framework_runners/*.py`) |
| `optimal_query_strategies/` | Oracle strategy implementations (see `optimal_query_strategies/*.py`) |
| `metrics/` | Metric computation modules (see `metrics/base_metric.py` and `metrics/Standard_ML_Metrics.py`) |
| `resources/` | Configuration and templates (see `resources/exp_config.yaml`, `resources/data_types.py`) |
| `scripts/` | Utility, conversion, and maintenance scripts (see `scripts/` directory) |
| `eva_scripts/` | Evaluation, visualization, and paper figure scripts (see `eva_scripts/` directory) |
| `misc/` | Shared utilities: config, logging, helpers (see `misc/config.py`, `misc/helpers.py`) |

### Deprecated

| Directory | Status |
|-----------|--------|
| `analyse_results/` | **Deprecated / not used.** Prefer `eva_scripts/` for analysis. |

## License

AGPL-3.0. See [LICENSE](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/LICENSE) for details.

---

## Documentation QA Checklist

This section helps maintain documentation quality and accuracy.

### Building Docs Locally

```bash
# Install MkDocs and dependencies
pip install mkdocs-material

# Serve docs locally (with live reload)
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Validating Mermaid Rendering

Mermaid diagrams are configured using **Approach B: mkdocs-material's Mermaid support via extra_javascript** in `mkdocs.yml`:

1. **Check configuration**: Verify `mkdocs.yml` has:
   ```yaml
   markdown_extensions:
     - pymdownx.superfences:
         custom_fences:
           - name: mermaid
             class: mermaid
             format: !!python/name:pymdownx.superfences.fence_div_format
   
   extra_javascript:
     - https://unpkg.com/mermaid@10/dist/mermaid.min.js
   ```

2. **Test locally**: Run `mkdocs serve` and check that diagrams render as flowcharts/graphs (not raw text)

3. **Valid syntax**: Mermaid blocks must use triple-backtick fences with `mermaid` language:
   ````markdown
   ```mermaid
   flowchart TD
       A --> B
   ```
   ````

### Spotting Stale or Unsafe Claims

When reviewing documentation, verify claims about code behavior with source code:

**Code pointer format**: `(source: path/to/file.py::ClassName.method_name)` or `(see path/to/file.py)`

**Red flags requiring verification**:

| Claim Type | Example | How to Verify |
|------------|---------|---------------|
| Default values | "default batch size is 10" | Check `misc/config.py::Config` class attributes or `resources/exp_config.yaml` |
| Output paths | "Results saved to `OUTPUT_PATH/metrics/`" | Check script that writes the file (e.g., `02_run_experiment.py`) |
| File formats | "CSV with columns X, Y, Z" | Check actual output generation code or `pandas.to_csv()` calls |
| Behavior claims | "Automatically resumes from checkpoint" | Check script logic for resume/checkpoint code |
| Config keys | "`EXP_GRID_DATASET` controls datasets" | Check `resources/data_types.py` enums and `misc/config.py` |

**Marking unverified claims**: If you cannot find source code support, add `TODO(verify):` prefix:

```markdown
TODO(verify): Default timeout is 300 seconds per query.
```

**Finding source code**:

- Configuration: `misc/config.py`, `resources/exp_config.yaml`, `resources/data_types.py`
- Pipeline scripts: `00_download_datasets.py` through `07b_create_results_without_flask.py`
- Output formats: Search for `to_csv()`, `to_parquet()`, file write operations
- Behavior: Read script main logic and helper functions in `misc/helpers.py`
