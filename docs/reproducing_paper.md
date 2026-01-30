# Reproducing the Paper Results

This document provides the complete workflow used to generate the results archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862), which correspond to the findings in the paper [arXiv:2506.03817](https://arxiv.org/abs/2506.03817) "Survey of Active Learning Hyperparameters".

## Overview

The OPARA archive (`full_exp_jan.zip`) was generated using the `full_exp_jan` experiment configuration defined in `resources/exp_config.yaml`. This document details:

1. The exact configuration used
2. The complete command sequence
3. Post-processing scripts to compute derived metrics
4. Evaluation scripts to generate paper figures

---

## Configuration: `full_exp_jan`

The archived data was generated using the following hyperparameter grid:

### Datasets (92 total)

```yaml
EXP_GRID_DATASET:
  - Bioresponse
  - GesturePhaseSegmentationProcessed
  - Internet-Advertisements
  - Iris
  - MiceProtein
  - PenDigits
  - ac-inflam
  - analcatdata_authorship
  - analcatdata_dmft
  - annthyroid
  - appendicitis
  - balance-scale
  - banana
  - banknote-authentication
  - blood
  - breast_cancer_wisconsin_original
  - car
  - cardiotocography
  - churn
  - climate-model-simulation-crashes
  - cmc
  - cnae-9
  - credit-g
  - cylinder
  - d31
  - diabetes
  - dresses-sales
  - dwtc
  - ex8a
  - ex8b
  - fertility
  - first-order-theorem-proving
  - gaussian_balanced
  - gaussian_unbalanced
  - glass
  - haberman
  - har
  - hepatitis
  - hill-valley
  - ilpd
  - ionosphere
  - isolet
  - kc1
  - kc2
  - kr-vs-kp
  - madelon
  - mammographic
  - mfeat-factors
  - mfeat-fourier
  - mfeat-karhunen
  - mfeat-morphological
  - mfeat-pixel
  - mfeat-zernike
  - molecular-biology
  - musk1
  - musk2
  - optdigits
  - ozone-level
  - page-blocks
  - parkinsons
  - pc1
  - pc3
  - pc4
  - phishing
  - phoneme
  - planning-relax
  - r15
  - ringnorm
  - satimage
  - scale
  - seeds
  - segment
  - semeion
  - sonar
  - soybean
  - spambase
  - splice_junction
  - statlog_australian
  - statlog_heart
  - steel-plates-fault
  - texture
  - thyroid
  - twonorm
  - vowel
  - wall-robot-navigation
  - waveform
  - wbc
  - wdbc
  - wilt
  - wine_origin
  - xor_2x2
  - xor_2x2_rotated
  - xor_4x4
```

### AL Strategies (28 total)

```yaml
EXP_GRID_STRATEGY:
  # ALiPy strategies
  - ALIPY_RANDOM
  - ALIPY_UNCERTAINTY_LC
  - ALIPY_UNCERTAINTY_MM
  - ALIPY_UNCERTAINTY_ENTROPY
  - ALIPY_GRAPH_DENSITY
  - ALIPY_CORESET_GREEDY
  - ALIPY_DENSITY_WEIGHTED
  
  # Optimal (oracle) strategies
  - OPTIMAL_GREEDY_10
  - OPTIMAL_GREEDY_20
  
  # libact strategies
  - LIBACT_UNCERTAINTY_LC
  - LIBACT_UNCERTAINTY_SM
  - LIBACT_UNCERTAINTY_ENT
  - LIBACT_DWUS
  - LIBACT_QUIRE
  
  # small-text strategies
  - SMALLTEXT_LEASTCONFIDENCE
  - SMALLTEXT_PREDICTIONENTROPY
  - SMALLTEXT_BREAKINGTIES
  - SMALLTEXT_EMBEDDINGKMEANS
  - SMALLTEXT_GREEDYCORESET
  - SMALLTEXT_LIGHTWEIGHTCORESET
  - SMALLTEXT_CONTRASTIVEAL
  - SMALLTEXT_RANDOM
  
  # scikit-activeml strategies
  - SKACTIVEML_QBC
  - SKACTIVEML_US_MARGIN
  - SKACTIVEML_US_LC
  - SKACTIVEML_US_ENTROPY
  - SKACTIVEML_EXPECTED_AVERAGE_PRECISION
  - SKACTIVEML_COST_EMBEDDING
  - SKACTIVEML_DAL
  - SKACTIVEML_QBC_VOTE_ENTROPY
  - SKACTIVEML_QUIRE
```

### Other Hyperparameters

The paper defines the hyperparameter grid as the Cartesian product **𝕊 × 𝔻 × 𝕋 × 𝕀 × 𝔹 × 𝕃**.

```yaml
# Metrics to record (𝕄 in paper)
METRICS:
  - Predicted_Samples
  - Selected_Indices
  - Standard_ML_Metrics
  - Timing_Metrics

# Random seed (fixed for reproducibility)
EXP_GRID_RANDOM_SEED: [0]

# Initial labeled set variations (𝕀 in paper - 20 different start sets)
# Note: [0-19] is OGAL shorthand syntax that expands to [0, 1, 2, ..., 19]
EXP_GRID_START_POINT: [0-19]

# Number of AL cycles (c in paper - 100 iterations)
EXP_GRID_NUM_QUERIES: [100]

# Batch sizes to evaluate (𝔹 in paper)
EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100]

# Learner models (𝕃 in paper)
EXP_GRID_LEARNER_MODEL: [MLP, RBF_SVM, RF]

# Train/test split buckets (𝕋 in paper - 5 different splits)
# Note: [0-4] is OGAL shorthand syntax that expands to [0, 1, 2, 3, 4]
EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0-4]
```

!!! note "OGAL YAML Shorthand"
    The notation `[0-19]` is OGAL-specific shorthand that automatically expands to `[0, 1, 2, ..., 19]`. This is parsed by the Config class when loading experiment configurations.

### Total Hyperparameter Combinations

Using the paper's notation:

```
|𝔻| × |𝕊| × |𝕀| × |𝔹| × |𝕃| × |𝕋|
= 92 datasets × 28 strategies × 20 start sets × 6 batch sizes × 3 learners × 5 splits
= 4,636,800 experiments
```

---

## Step-by-Step Reproduction

### Phase 1: Setup

```bash
# 1. Create environment
conda create --name al_olympics_env --file conda-linux-64.lock
conda activate al_olympics_env
poetry install

# 2. Configure paths in .server_access_credentials.cfg
```

Create `.server_access_credentials.cfg` with at minimum:

```ini
[HPC]
HPC_DATASETS_PATH = /path/to/hpc/datasets/
HPC_OUTPUT_PATH = /path/to/hpc/output/

[LOCAL]
LOCAL_DATASETS_PATH = /path/to/local/datasets/
LOCAL_OUTPUT_PATH = /path/to/local/output/
```

See [Configuration](configuration.md) for a complete list of configuration fields.

### Phase 2: Download Datasets

```bash
python 00_download_datasets.py
```

### Phase 3: Create Workload

```bash
# Using the YAML configuration
python 01_create_workload.py --USE_EXP_YAML full_exp_jan
```

This generates:
- `OUTPUT_PATH/full_exp_jan/01_workload.csv` (4.6M rows)
- `OUTPUT_PATH/full_exp_jan/00_config.yaml`
- `OUTPUT_PATH/full_exp_jan/02_slurm.slurm`

### Phase 4: Run Experiments on HPC

```bash
# Submit SLURM job array
sbatch OUTPUT_PATH/full_exp_jan/02_slurm.slurm

# Monitor progress
watch -n 60 'wc -l OUTPUT_PATH/full_exp_jan/05_done_workload.csv'
```

**Resource requirements:**
- ~3.6 million CPU hours
- Runtime per experiment: seconds to minutes (5-minute timeout per AL cycle)
- Storage: several terabytes for raw results

### Phase 5: Post-Processing

After experiments complete, compute derived metrics:

#### 5a. Calculate Dataset Categorizations

```bash
python 03_calculate_dataset_categorizations.py \
    --EXP_TITLE full_exp_jan \
    --SAMPLES_CATEGORIZER _ALL \
    --EVA_MODE create

# For HPC execution
sbatch OUTPUT_PATH/full_exp_jan/workloads/DATASET_CATEGORIZATIONS/02_slurm.slurm

# Or local execution
python 03_calculate_dataset_categorizations.py \
    --EXP_TITLE full_exp_jan \
    --SAMPLES_CATEGORIZER _ALL \
    --EVA_MODE local
```

#### 5b. Calculate Advanced Metrics

```bash
python 04_calculate_advanced_metrics.py \
    --EXP_TITLE full_exp_jan \
    --COMPUTED_METRICS _ALL \
    --EVA_MODE create

# For HPC execution
sbatch OUTPUT_PATH/full_exp_jan/workloads/advanced_metrics/02_slurm.slurm

# Or local execution
python 04_calculate_advanced_metrics.py \
    --EXP_TITLE full_exp_jan \
    --COMPUTED_METRICS _ALL \
    --EVA_MODE local
```

#### 5c. Analyze Workload Completion

```bash
python 05_analyze_partially_run_workload.py --EXP_TITLE full_exp_jan
```

---

## Phase 6: Evaluation Scripts

The following scripts generate the paper's figures and tables:

### Basic Analysis

```bash
# Learning curves
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan

# Runtime analysis
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan

# Basic metrics correlation
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan

# AUC metrics correlation
python -m eva_scripts.auc_metric_correlation --EXP_TITLE full_exp_jan
```

### Hyperparameter Influence Analysis

```bash
# Single hyperparameter evaluation (indices)
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan

# Single hyperparameter evaluation (metrics)
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan

# Leaderboard hyperparameter influence analysis
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
```

### Leaderboard Scenarios

```bash
# Dataset scenarios
python -m eva_scripts.leaderboard_scenarios \
    --EXP_TITLE full_exp_jan \
    --EVA_MODE local \
    --SCENARIOS dataset_scenario

# Start point scenarios
python -m eva_scripts.leaderboard_scenarios \
    --EXP_TITLE full_exp_jan \
    --EVA_MODE local \
    --SCENARIOS start_point_scenario

# Minimum hyperparameter scenarios
python -m eva_scripts.leaderboard_scenarios \
    --EXP_TITLE full_exp_jan \
    --EVA_MODE create \
    --SCENARIOS min_hyper

python -m eva_scripts.leaderboard_scenarios \
    --EXP_TITLE full_exp_jan \
    --EVA_MODE local \
    --SCENARIOS min_hyper
```

### Dataset-Dependent Analysis

```bash
# Calculate random strategy ramp-up slope per dataset
python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE full_exp_jan

# Create AUC time series for selected indices
python -m scripts.create_auc_selected_ts --EXP_TITLE full_exp_jan
```

### Workload Reduction Analysis

The workload reduction analysis studies how experiment results change when using fewer hyperparameter combinations. This is used for the paper's RQ2 about minimal hyperparameter grids.

```bash
# Analyze workload reduction at different thresholds
# The nested loop runs multiple iterations per threshold to ensure statistical robustness
# - create: generates the reduced workload definition
# - local: computes leaderboard rankings on reduced workload
# - reduce: stores results indexed by worker for later aggregation

for threshold in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99; do
    for i in $(seq 0 50); do
        # Create workload for this iteration
        python -m eva_scripts.workload_reduction \
            --EXP_TITLE full_exp_jan \
            --EVA_WORKLOAD_REDUCTION_THRESHOLD $threshold \
            --EVA_MODE create \
            --WORKER_INDEX $i
        
        # Compute rankings on reduced workload
        python -m eva_scripts.workload_reduction \
            --EXP_TITLE full_exp_jan \
            --EVA_WORKLOAD_REDUCTION_THRESHOLD $threshold \
            --EVA_MODE local
        
        # Store iteration results
        python -m eva_scripts.workload_reduction \
            --EXP_TITLE full_exp_jan \
            --EVA_WORKLOAD_REDUCTION_THRESHOLD $threshold \
            --EVA_MODE reduce \
            --WORKER_INDEX $i
    done
    
    # Move results to threshold-specific directory
    mv OUTPUT_PATH/full_exp_jan/workloads/workload_reduction \
       OUTPUT_PATH/full_exp_jan/workloads/workload_reduction_$(echo $threshold | tr -d '.')
done
```

---

## Complete Command Sequence

Here is the complete command sequence used to generate the OPARA archive (as reflected in `test.sh`):

```bash
#!/bin/bash

# === MAIN PIPELINE ===

# Step 0: Download datasets
python 00_download_datasets.py

# Step 1: Create workload
python 01_create_workload.py --USE_EXP_YAML full_exp_jan

# Step 2: Run experiments (HPC)
sbatch OUTPUT_PATH/full_exp_jan/02_slurm.slurm

# Step 3: Calculate dataset categorizations
python 03_calculate_dataset_categorizations.py --EXP_TITLE full_exp_jan --SAMPLES_CATEGORIZER _ALL --EVA_MODE local

# Step 4: Calculate advanced metrics
python 04_calculate_advanced_metrics.py --EXP_TITLE full_exp_jan --COMPUTED_METRICS _ALL --EVA_MODE local

# Step 5: Analyze completion
python 05_analyze_partially_run_workload.py --EXP_TITLE full_exp_jan

# === EVALUATION SCRIPTS ===

# Learning curves and basic plots
python -m eva_scripts.single_learning_curve_example --EXP_TITLE full_exp_jan
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.auc_metric_correlation --EXP_TITLE full_exp_jan

# Hyperparameter analysis
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan
python -m eva_scripts.single_hyperparameter_evaluation_metric --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan

# Dataset-specific analysis
python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE full_exp_jan
python -m scripts.create_auc_selected_ts --EXP_TITLE full_exp_jan

# Leaderboard scenarios
python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS dataset_scenario
python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS start_point_scenario
python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS min_hyper
```

---

## Outputs Produced

### Raw Experiment Results

Location: `OUTPUT_PATH/full_exp_jan/<STRATEGY>/<DATASET>/`

| File | Description |
|------|-------------|
| `accuracy.csv` | Per-cycle accuracy |
| `weighted_f1-score.csv` | Per-cycle weighted F1 |
| `macro_f1-score.csv` | Per-cycle macro F1 |
| `query_selection_time.csv` | Strategy query time |
| `selected_indices.csv` | Selected sample indices |

### Advanced Metrics

Location: `OUTPUT_PATH/full_exp_jan/<STRATEGY>/<DATASET>/`

| File | Description |
|------|-------------|
| `full_auc_*.csv` | Full AUC of learning curve |
| `first_5_*.csv` | Mean of first 5 iterations |
| `last_5_*.csv` | Mean of last 5 iterations |
| `final_value_*.csv` | Final iteration value |
| `ramp_up_auc_*.csv` | AUC of ramp-up phase |
| `plateau_auc_*.csv` | AUC of plateau phase |

### Dataset Categorizations

Location: `OUTPUT_PATH/full_exp_jan/<STRATEGY>/<DATASET>/`

| File | Description |
|------|-------------|
| `AVERAGE_UNCERTAINTY.csv` | Per-sample uncertainty |
| `CLOSENESS_TO_DECISION_BOUNDARY.csv` | Distance to boundary |
| `CLOSENESS_TO_CLUSTER_CENTER.csv` | Distance to center |
| `COUNT_WRONG_CLASSIFICATIONS.csv` | Misclassification count |
| `MELTING_POT_REGION.csv` | Mixed-class region |
| `OUTLIERNESS.csv` | Outlier score |
| `REGION_DENSITY.csv` | Local density |
| `SWITCHES_CLASS_OFTEN.csv` | Prediction instability |

### Evaluation Outputs

Location: `OUTPUT_PATH/full_exp_jan/plots/`

| Directory | Contents |
|-----------|----------|
| `single_learning_curve/` | Learning curve examples |
| `runtime/` | Runtime analysis |
| `basic_metrics/` | Metric correlations |
| `final_leaderboard/` | Strategy rankings |
| `leaderboard_single_hyperparameter_influence/` | Hyperparameter sensitivity |

---

## Verifying Reproduction

To verify your reproduction matches the archive:

```python
import pandas as pd

# Load your results
your_done = pd.read_csv("OUTPUT_PATH/full_exp_jan/05_done_workload.csv")
print(f"Completed experiments: {len(your_done)}")

# Should be approximately 4.5M+ (some fail due to timeouts/errors)

# Load a specific metric file
accuracy = pd.read_csv(
    "OUTPUT_PATH/full_exp_jan/ALIPY_RANDOM/Iris/accuracy.csv"
)
print(f"Accuracy rows: {len(accuracy)}")

# Should have entries for each hyperparameter combination for this dataset/strategy
```

---

## Troubleshooting

### Missing Experiments

```bash
# Find missing experiment IDs
python -m scripts.find_missing_exp_ids_in_metric_files --EXP_TITLE full_exp_jan

# Rerun broken experiments
python -m scripts.rerun_broken_experiments --EXP_TITLE full_exp_jan
```

### OOM Issues

```bash
# Fix OOM workload (remove completed from retry list)
python -m scripts.fix_oom_workload --EXP_TITLE full_exp_jan
```

### Data Validation

```bash
# Check for broken files
python -m scripts.find_broken_file --EXP_TITLE full_exp_jan

# Validate experiment IDs
python -m scripts.check_if_exp_ids_are_present --EXP_TITLE full_exp_jan
```

---

## Subset Reproduction

If you cannot run the full 4.6M experiments, you can reproduce a subset:

### Minimal Subset

```yaml
# Create a smaller configuration in resources/exp_config.yaml
mini_reproduction:
  EXP_GRID_DATASET: [Iris, wine_origin, diabetes]
  EXP_GRID_STRATEGY: [ALIPY_RANDOM, ALIPY_UNCERTAINTY_LC]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_START_POINT: [0-4]
  EXP_GRID_NUM_QUERIES: [50]
  EXP_GRID_BATCH_SIZE: [1, 10]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
  METRICS: [Standard_ML_Metrics, Timing_Metrics]
```

Then run:

```bash
python 01_create_workload.py --USE_EXP_YAML mini_reproduction
python 02_run_experiment.py --EXP_TITLE mini_reproduction --WORKER_INDEX 0
```

### Using Archived Results

You can also use the archived results directly for evaluation:

```bash
# Download and extract full_exp_jan.zip from OPARA
unzip full_exp_jan.zip

# Configure OGAL to point to extracted data
# In .server_access_credentials.cfg:
# LOCAL_OUTPUT_PATH = /path/to/full_exp_jan/

# Run evaluation scripts directly on archived data
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
```

---

## References

- **Paper**: [arXiv:2506.03817](https://arxiv.org/abs/2506.03817) - "Survey of Active Learning Hyperparameters"
- **Code Repository**: [jgonsior/olympic-games-of-active-learning](https://github.com/jgonsior/olympic-games-of-active-learning)
- **Archived Data**: [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)
