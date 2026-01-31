# Start Here

**OGAL (Olympic Games of Active Learning)** is the largest Active Learning benchmarking study to date, with **4.6 million hyperparameter combinations** already computed and archived.

!!! success "Skip 3.6 million CPU hours"
    The complete experiment results are archived at **[DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)**. You can analyze this data without rerunning experiments.

---

## What Do You Want to Do?

<div class="grid cards" markdown>

-   :material-magnify-scan:{ .lg .middle } **Analyze the Dataset**

    ---

    Use the published 4.6M experiment results for your own research — stopping criteria, meta-learning, strategy recommendations.

    [:octicons-arrow-right-24: Analyze the dataset](analyze_dataset.md)

-   :material-plus-circle:{ .lg .middle } **Add Your Results**

    ---

    Contribute new experiments (strategies, datasets, learners) to the shared benchmark.

    [:octicons-arrow-right-24: Add your results](add_results.md)

-   :material-cog:{ .lg .middle } **Run the Benchmark**

    ---

    Execute experiments locally or at HPC scale with SLURM.

    [:octicons-arrow-right-24: Reference → Runbook](reference/runbook.md)

</div>

---

## The OPARA Dataset

| Fact | Value |
|------|-------|
| **Experiments** | 4.6+ million hyperparameter combinations |
| **AL Strategies** | 28 from 5 frameworks (ALiPy, libact, small-text, scikit-activeml, playground) |
| **Datasets** | 92 classification tasks |
| **Learners** | Random Forest, MLP, SVM |
| **Compute invested** | ~3.6 million CPU hours |
| **Archive** | [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862) |
| **Paper** | [arXiv:2506.03817](https://arxiv.org/abs/2506.03817) |

### What's in the Archive

```
full_exp_jan/
├── <STRATEGY>/<DATASET>/
│   ├── accuracy.csv.xz           # Per-cycle accuracy
│   ├── weighted_f1-score.csv.xz  # Per-cycle F1
│   ├── selected_indices.csv.xz   # Queried sample indices
│   ├── full_auc_*.csv.xz         # Aggregated AUC metrics
│   └── ...
├── 05_done_workload.csv          # 4.6M completed experiments
└── 01_workload.csv               # Full hyperparameter grid
```

---

## Paper Terminology

The paper ([arXiv:2506.03817](https://arxiv.org/abs/2506.03817)) defines an AL experiment as **E = (𝒮, D, 𝒯, ℐ, M, b, c, ℒ)**:

| Symbol | Term | Values in Archive |
|--------|------|-------------------|
| 𝕊 | AL Strategy | 28 strategies |
| 𝔻 | Dataset | 92 datasets |
| 𝕃 | Learner Model | RF, MLP, SVM |
| 𝔹 | Batch Size | 1, 5, 10, 20, 50, 100 |
| 𝕋 | Train-Test Split | 5 splits per dataset |
| 𝕀 | Initial Start Set | 20 start sets per split |
| c | AL Cycles | 100 iterations |

---

## Quick Start: Analyze Without Rerunning

**Goal:** Generate a strategy leaderboard from pre-computed results—no experiments needed.

**Inputs:**

| Input | Source |
|-------|--------|
| Archived results (ZIP) | [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862) |

??? note "Convenience link (may change)"
    Direct download (not guaranteed stable):
    ```
    https://opara.zih.tu-dresden.de/xmlui/bitstream/handle/123456789/5678/full_exp_jan.zip
    ```

**Run:**

```bash
# 1. Download and extract archived results
wget <URL_FROM_DOI_LANDING_PAGE>
unzip full_exp_jan.zip -d /path/to/results/

# 2. Setup OGAL environment
conda create --name ogal --file conda-linux-64.lock
conda activate ogal
poetry install

# 3. Configure paths (define OGAL_OUTPUT once)
export OGAL_OUTPUT=/path/to/results
cat > .server_access_credentials.cfg << EOF
[LOCAL]
OUTPUT_PATH=${OGAL_OUTPUT}
DATASETS_PATH=/path/to/datasets
EOF

# 4. Generate leaderboard from archived data
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

**You should see:**

| Artifact | Location |
|----------|----------|
| Leaderboard heatmap | `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.jpg` |
| Leaderboard data | `plots/final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet` |

!!! tip "Sanity check"
    If the leaderboard files do not appear, see [Reference → Eva Scripts Catalog → final_leaderboard.py](reference/eva_scripts_catalog.md#final_leaderboardpy).

See [Analyze the Dataset](analyze_dataset.md) for research starter analyses.

---

## Deprecated

!!! warning "analyse_results/"
    The `analyse_results/` directory is **deprecated**. Use `eva_scripts/` for all analysis.

---

## Navigation

| Section | Purpose |
|---------|---------|
| [Analyze the Dataset](analyze_dataset.md) | Research using archived data |
| [Add Your Results](add_results.md) | Contribute new experiments |
| [Reference](reference/runbook.md) | Pipeline, HPC, schemas, catalogs |
| [Contributing](contributing.md) | Development setup |

---

## Docs Maintenance

??? info "Where should new information go?"
    
    | Content Type | Location |
    |--------------|----------|
    | Research workflows with existing data | [Analyze the Dataset](analyze_dataset.md) |
    | Adding new experiments/strategies | [Add Your Results](add_results.md) |
    | Pipeline details, HPC, schemas | [Reference](reference/runbook.md) |
    | Script I/O specifications | [Reference → Eva Scripts](reference/eva_scripts_catalog.md) |
    | Mathematical definitions | [Reference → Correlations](reference/correlations_paper_to_code.md) |

??? info "Building docs locally"
    ```bash
    pip install mkdocs-material mkdocs-mermaid2-plugin
    mkdocs serve
    # Open http://127.0.0.1:8000
    ```
