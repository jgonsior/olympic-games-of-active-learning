# Data Enrichment Protocol

This document defines the strict protocol for adding new experiment results to the OGAL shared dataset. Follow these steps exactly to ensure data integrity and reproducibility.

!!! warning "Before You Begin"
    - Read [Results Format](results_format.md) to understand the output schema
    - Read [Evaluation Pipeline](evaluation_pipeline.md) to understand the processing chain
    - Ensure your experiments use the same schema version as existing data

---

## 1. Run Identity (Primary Key)

### 1.1 Definition

Each experiment has a unique identity defined by the combination of hyperparameters. This serves as the primary key for deduplication and merge operations.

**Run Identity = (EXP_DATASET, EXP_STRATEGY, EXP_LEARNER_MODEL, EXP_BATCH_SIZE, EXP_START_POINT, EXP_TRAIN_TEST_BUCKET_SIZE, EXP_RANDOM_SEED)**

Additionally, each run has a unique `EXP_UNIQUE_ID` integer assigned during workload creation.

(source: `01_create_workload.py::_generate_exp_param_grid`, lines 40-98)

### 1.2 Where Run Identity is Stored

| Location | Format | Source |
|----------|--------|--------|
| `01_workload.csv` | CSV with all hyperparameter columns | `01_create_workload.py` |
| `05_done_workload.csv` | Subset of completed runs | `framework_runners/base_runner.py` |
| `EXP_UNIQUE_ID` column | Integer primary key in all metric files | `metrics/base_metric.py` |

### 1.3 Computing Run Identity in Code

```python
# From misc/config.py - these fields define the identity
identity_fields = [
    'EXP_DATASET',      # DATASET enum value (int)
    'EXP_STRATEGY',     # AL_STRATEGY enum value (int)
    'EXP_LEARNER_MODEL',# LEARNER_MODEL enum value (int)
    'EXP_BATCH_SIZE',   # Query batch size (int)
    'EXP_START_POINT',  # Initial labeled set index (int)
    'EXP_TRAIN_TEST_BUCKET_SIZE',  # Train/test split (int)
    'EXP_RANDOM_SEED',  # Random seed (int)
]
```

(source: `misc/config.py::Config` class attributes)

---

## 2. Results Schema Contract

### 2.1 Required Fields (MUST HAVE)

Every valid results bundle MUST contain:

| File | Required Columns | Type |
|------|-----------------|------|
| `05_done_workload.csv` | All identity fields + `EXP_UNIQUE_ID`, `EXP_NUM_QUERIES` | int |
| `<STRATEGY>/<DATASET>/accuracy.csv.xz` | `EXP_UNIQUE_ID`, `0`, `1`, ..., `N` | int, float... |
| `<STRATEGY>/<DATASET>/weighted_f1-score.csv.xz` | Same as above | int, float... |

### 2.2 Optional Fields

| File | Description | When Required |
|------|-------------|---------------|
| `05_failed_workloads.csv` | Failed experiments | If any failures |
| `05_started_oom_workloads.csv` | OOM experiments | If any OOM |
| `selected_indices.csv.xz` | Queried sample indices | For index analysis |
| `query_selection_time.csv.xz` | Timing data | For runtime analysis |
| `y_pred_*.parquet` | Predictions | For prediction analysis |

### 2.3 Schema Versioning

OGAL does not have formal schema versioning. Compatibility is maintained by:

1. **Column consistency**: Metric files must have `EXP_UNIQUE_ID` as first column
2. **Enum consistency**: Strategy/Dataset/Model values must match `resources/data_types.py`
3. **Path consistency**: Files must follow `<STRATEGY_NAME>/<DATASET_NAME>/<metric>.csv.xz`

**Proposed Schema Version**: Include in `00_config.yaml`:

```yaml
SCHEMA_VERSION: "1.0"  # Add this field for future compatibility
```

---

## 3. Ingestion Protocol

### Step 3.1: Prepare Raw Outputs

Place your raw experiment outputs in a separate directory:

```
NEW_RESULTS_ROOT/
├── 00_config.yaml              # Your experiment configuration
├── 01_workload.csv             # Full workload definition
├── 05_done_workload.csv        # Completed experiments
├── 05_failed_workloads.csv     # Failed experiments (if any)
├── <STRATEGY_NAME>/
│   └── <DATASET_NAME>/
│       ├── accuracy.csv.xz
│       ├── weighted_f1-score.csv.xz
│       └── ...
```

### Step 3.2: Validate Schema

Run the validator script before merging:

```bash
python scripts/validate_results_schema.py --results_path /path/to/NEW_RESULTS_ROOT
```

The validator checks:
- Required columns present
- Types are reasonable (numeric where expected)
- No duplicate `EXP_UNIQUE_ID` within the bundle
- Consistent `EXP_UNIQUE_ID` across all metric files

See [Validation Step](#4-validation-step) for details.

### Step 3.3: Check for Duplicates Against Existing Data

Before merging, verify no duplicate run identities:

```bash
python scripts/validate_results_schema.py \
    --results_path /path/to/NEW_RESULTS_ROOT \
    --compare_with /path/to/EXISTING_RESULTS_ROOT
```

This will report any overlapping `EXP_UNIQUE_ID` values or duplicate hyperparameter combinations.

### Step 3.4: Run Post-Processing Scripts

After validation, generate derived artifacts:

```bash
# Set EXP_TITLE to your new results directory name
export NEW_EXP_TITLE="your_new_experiment"

# Step 3: Calculate dataset categorizations
python 03_calculate_dataset_categorizations.py \
    --EXP_TITLE $NEW_EXP_TITLE \
    --SAMPLES_CATEGORIZER _ALL \
    --EVA_MODE local

# Step 4: Calculate advanced metrics (AUC, etc.)
python 04_calculate_advanced_metrics.py \
    --EXP_TITLE $NEW_EXP_TITLE \
    --COMPUTED_METRICS _ALL \
    --EVA_MODE local

# Step 5: Analyze completion status
python 05_analyze_partially_run_workload.py --EXP_TITLE $NEW_EXP_TITLE
```

### Step 3.5: Merge Into Combined Dataset

**Option A: Separate Output Roots (Recommended)**

Keep results in separate directories with distinct `EXP_TITLE`:

```
OUTPUT_PATH/
├── original_experiment/     # Existing results
│   └── ...
├── new_experiment/          # Your new results
│   └── ...
```

Run eva_scripts separately and compare:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE original_experiment
python -m eva_scripts.final_leaderboard --EXP_TITLE new_experiment
```

**Option B: Merge Workloads (Advanced)**

Use the merge script to combine workloads:

```bash
python -m scripts.merge_two_workloads \
    --EXP_TITLE original_experiment \
    --SECOND_MERGE_PATH /path/to/new_experiment
```

(source: `scripts/merge_two_workloads.py`)

!!! warning "Merge Precautions"
    - Ensure no duplicate `EXP_UNIQUE_ID` values
    - Back up existing data before merging
    - Re-run post-processing after merge

### Step 3.6: Provenance Tagging

To track data provenance, add a `RUN_GROUP` or `PROVENANCE` field to your `00_config.yaml`:

```yaml
# In 00_config.yaml
PROVENANCE:
  run_group: "experiment_v2_jan2025"
  researcher: "your_name"
  date: "2025-01-31"
  git_commit: "abc123def"
  description: "Extended hyperparameter grid with additional batch sizes"
```

---

## 4. Validation Step

### 4.1 Validator Script

Use `scripts/validate_results_schema.py` to verify data integrity:

```bash
# Basic validation
python scripts/validate_results_schema.py --results_path /path/to/results

# Compare with existing data
python scripts/validate_results_schema.py \
    --results_path /path/to/new_results \
    --compare_with /path/to/existing_results

# Strict mode (fail on warnings)
python scripts/validate_results_schema.py \
    --results_path /path/to/results \
    --strict
```

### 4.2 What the Validator Checks

| Check | Severity | Description |
|-------|----------|-------------|
| Required columns present | ERROR | `EXP_UNIQUE_ID` must exist in all metric files |
| Types reasonable | WARNING | Numeric columns should be numeric |
| No duplicate primary keys | ERROR | `EXP_UNIQUE_ID` must be unique within workload |
| Consistent config metadata | WARNING | `00_config.yaml` should exist and be valid YAML |
| Cross-file consistency | ERROR | `EXP_UNIQUE_ID` in metrics must exist in `05_done_workload.csv` |
| No overlap with comparison | WARNING | When comparing, report duplicate identities |

### 4.3 Interpreting Failures

| Exit Code | Meaning | Action |
|-----------|---------|--------|
| 0 | All checks passed | Safe to proceed |
| 1 | Errors found | Fix errors before proceeding |
| 2 | Warnings only | Review warnings, proceed with caution |

**Example Output:**

```
=== OGAL Results Schema Validator ===
Results path: /path/to/results

[CHECK] Required files...
  ✓ 05_done_workload.csv exists
  ✓ 00_config.yaml exists

[CHECK] Workload schema...
  ✓ EXP_UNIQUE_ID column present
  ✓ All identity columns present
  ✓ 1,234 unique experiments

[CHECK] Metric files...
  ✓ 28 strategy directories found
  ✓ Checking ALIPY_RANDOM/Iris/accuracy.csv.xz... OK
  ...

[CHECK] Cross-file consistency...
  ✓ All EXP_UNIQUE_ID values in metrics exist in workload

[CHECK] Duplicate primary keys...
  ✓ No duplicates found

=== VALIDATION PASSED ===
```

---

## 5. Best Practices

### 5.1 Before Running New Experiments

1. **Use consistent configuration**: Copy `resources/exp_config.yaml` and modify
2. **Use separate `EXP_TITLE`**: Never overwrite existing experiment directories
3. **Document your changes**: Update `00_config.yaml` with provenance info

### 5.2 During Experiments

1. **Monitor progress**: Check `05_done_workload.csv` periodically
2. **Track failures**: Review `05_failed_workloads.csv` for systematic issues
3. **Back up incrementally**: Copy results to backup location periodically

### 5.3 After Experiments

1. **Validate immediately**: Run validator before any analysis
2. **Run post-processing**: Generate derived metrics (Steps 03, 04)
3. **Compare with baseline**: Run eva_scripts and compare with existing results
4. **Document findings**: Note any anomalies or differences

### 5.4 Sharing Results

When sharing results with others:

1. Include `00_config.yaml` with full configuration
2. Include `05_done_workload.csv` for reproducibility
3. Document any custom strategies or modifications
4. Provide git commit hash of code used
5. Run validator and include validation report

---

## 6. Troubleshooting

### Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| Duplicate `EXP_UNIQUE_ID` | Re-running without clearing | Use fresh workload or filter duplicates |
| Missing metric files | Early termination or OOM | Check `05_failed_workloads.csv`, rerun |
| Type mismatch | Corrupted CSV | Run `scripts/find_broken_file.py` |
| Enum value not found | Outdated `data_types.py` | Update enums or use numeric values |

### Recovery Steps

**If validation fails with duplicates:**

```bash
# Remove duplicate entries
python -m scripts.remove_duplicated_exp_ids --EXP_TITLE your_experiment
```

**If metric files are corrupted:**

```bash
# Find broken files
python -m scripts.find_broken_file --EXP_TITLE your_experiment

# Fix or remove broken files manually
```

**If workload is incomplete:**

```bash
# Identify missing experiments
python -m scripts.find_missing_exp_ids_in_metric_files --EXP_TITLE your_experiment

# Rerun missing experiments
python -m scripts.rerun_missing_exp_ids --EXP_TITLE your_experiment
```
