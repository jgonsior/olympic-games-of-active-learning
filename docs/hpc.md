# HPC Setup Guide

This document is the **authoritative HPC runbook** for running OGAL experiments on High-Performance Computing clusters using SLURM job arrays.

!!! tip "Quick Navigation"
    - [Smoke Test Path](#minimal-local-smoke-test) - Verify setup locally first
    - [SLURM Array Sizing](#sizing-slurm-arrays) - Calculate array indices from workload
    - [Resume Behavior](#resume-and-recovery) - Understand completion tracking
    - [Troubleshooting](#hpc-troubleshooting-checklist) - Common issues and fixes

## Overview

OGAL is designed for massive parallelization on HPC clusters. The key mechanism is:

1. **Workload partitioning**: `01_create_workload.py` generates a CSV with thousands/millions of experiment configurations
2. **Worker sharding**: Each SLURM job array task runs `02_run_experiment.py` with a different `WORKER_INDEX`
3. **Result aggregation**: Results are written to shared storage and tracked in `05_done_workload.csv`

---

## Minimal Local Smoke Test

Before deploying to HPC, verify your setup works locally:

```bash
# 1. Create a tiny workload (should complete in minutes)
# Add this to resources/exp_config.yaml:
# smoke_test:
#   EXP_GRID_DATASET: [Iris]
#   EXP_GRID_STRATEGY: [ALIPY_RANDOM]
#   EXP_GRID_LEARNER_MODEL: [RF]
#   EXP_GRID_BATCH_SIZE: [5]
#   EXP_GRID_RANDOM_SEED: [0]
#   EXP_GRID_START_POINT: [0]
#   EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
#   EXP_GRID_NUM_QUERIES: [10]
#   METRICS: [Standard_ML_Metrics, Timing_Metrics]

python 01_create_workload.py --EXP_TITLE smoke_test

# 2. Run a single experiment locally
python 02_run_experiment.py --EXP_TITLE smoke_test --WORKER_INDEX 0

# 3. Verify outputs exist
ls -la OUTPUT_PATH/smoke_test/ALIPY_RANDOM/Iris/
# Should see: accuracy.csv, weighted_f1-score.csv, etc.

# 4. Verify completion tracking
cat OUTPUT_PATH/smoke_test/05_done_workload.csv
# Should have 1 row (plus header)
```

---

## Server Access Credentials

**File:** `.server_access_credentials.cfg`

This file configures paths for both local development and HPC execution. The `RUNNING_ENVIRONMENT` CLI flag (`local` or `hpc`) determines which section is used.

(source: `misc/config.py::Config._load_server_setup_from_file`)

### Safe Example Configuration

```ini
[HPC]
SSH_LOGIN=user@login.hpc.example.edu
WS_PATH=/data/workspace/al_olympics
DATASETS_PATH=/data/workspace/al_olympics/datasets
OUTPUT_PATH=/data/workspace/al_olympics/exp_results
SLURM_MAIL=your.email@example.edu
SLURM_PROJECT=your_project_account
CODE_PATH=/data/workspace/al_olympics/code
PYTHON_PATH=/data/workspace/al_olympics/conda-env/bin/python

[LOCAL]
DATASETS_PATH=/home/user/ogal/datasets
CODE_PATH=/home/user/ogal/code
OUTPUT_PATH=/home/user/ogal/exp_results
```

### Field Reference

| Field | Description | Used When |
|-------|-------------|-----------|
| `SSH_LOGIN` | SSH connection string for remote access | `sync_and_run.sh` script |
| `WS_PATH` | Base workspace path on HPC filesystem | SLURM templates |
| `DATASETS_PATH` | Path to datasets (should be on fast storage) | `RUNNING_ENVIRONMENT=hpc` |
| `OUTPUT_PATH` | Path for experiment outputs (shared storage) | `RUNNING_ENVIRONMENT=hpc` |
| `SLURM_MAIL` | Email for job notifications | SLURM templates |
| `SLURM_PROJECT` | SLURM account/project for allocation | SLURM templates |
| `CODE_PATH` | Path to OGAL repository on HPC | SLURM templates |
| `PYTHON_PATH` | Python interpreter path on HPC | SLURM templates |

### How Paths Are Selected

(source: `misc/config.py::Config._pathes_magic`, lines 204-211)

```python
if self.RUNNING_ENVIRONMENT == "local":
    self.OUTPUT_PATH = Path(self.LOCAL_OUTPUT_PATH)
    self.DATASETS_PATH = Path(self.LOCAL_DATASETS_PATH)
elif self.RUNNING_ENVIRONMENT == "hpc":
    self.OUTPUT_PATH = Path(self.HPC_OUTPUT_PATH)
    self.DATASETS_PATH = Path(self.HPC_DATASETS_PATH)
```

---

## WORKER_INDEX and Sharding Semantics

The `WORKER_INDEX` parameter is the core sharding mechanism that maps parallel workers to experiment configurations.

### Exact Implementation

(source: `misc/config.py::Config.load_workload`, lines 393-421)

```python
def load_workload(self) -> None:
    workload_df = pd.read_csv(
        self.WORKLOAD_FILE_PATH,
        header=0,
        index_col=None,
        skiprows=lambda x: x not in [0, self.WORKER_INDEX + 1],  # Read only header + target row
    )
    workload = workload_df.iloc[0].to_dict()
    # ... loads parameters from the single workload row
```

### How WORKER_INDEX Maps to Workload Rows

| WORKER_INDEX | CSV Row Read | Experiment |
|--------------|--------------|------------|
| 0 | Row 1 (after header) | First experiment configuration |
| 1 | Row 2 | Second experiment configuration |
| N | Row N+1 | (N+1)th experiment configuration |

**Key behaviors:**

- `WORKER_INDEX` is **0-indexed** 
- Each worker reads **exactly one row** from `01_workload.csv`
- The workload file is **shuffled** during creation to distribute long-running jobs (source: `01_create_workload.py`, line 94)

### Batching Mechanism

For very large workloads, OGAL supports batching multiple experiments per SLURM array task.

(source: `resources/slurm_templates/slurm_parallel.sh`, lines 20-25)

```bash
# Batching calculation in SLURM template
i=$(( {{ SLURM_OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ SLURM_ITERATIONS_PER_BATCH }} ))
end=$(($i+{{ SLURM_ITERATIONS_PER_BATCH }}))
for ((j = $i ; j < $end ; j++)); do
    ... python 02_run_experiment.py ... --WORKER_INDEX $j
done
```

| Parameter | Description | Default | Source |
|-----------|-------------|---------|--------|
| `SLURM_ITERATIONS_PER_BATCH` | Experiments per array task | 100 | `misc/config.py::Config.SLURM_ITERATIONS_PER_BATCH` |
| `SLURM_OFFSET` | Starting array index offset | 0 | `misc/config.py::Config.SLURM_OFFSET` |

**Example with batching (100 per batch):**

| SLURM_ARRAY_TASK_ID | WORKER_INDEX Range |
|---------------------|-------------------|
| 0 | 0-99 |
| 1 | 100-199 |
| 2 | 200-299 |

---

## Sizing SLURM Arrays

### Computing Array Size from Workload

(source: `01_create_workload.py::create_AL_experiment_slurm_files`, lines 321-341)

The array size is calculated as:

```python
array_end = int(workload_amount / config.SLURM_ITERATIONS_PER_BATCH)
# SLURM array: --array=0-{array_end}
```

### Recipe for Manual Array Sizing

```bash
# 1. Get workload length
WORKLOAD_LEN=$(wc -l < OUTPUT_PATH/your_experiment/01_workload.csv)
WORKLOAD_LEN=$((WORKLOAD_LEN - 1))  # Subtract header

# 2. Calculate array end index
BATCH_SIZE=100  # SLURM_ITERATIONS_PER_BATCH
ARRAY_END=$((WORKLOAD_LEN / BATCH_SIZE))

# 3. Use in SLURM
#SBATCH --array=0-${ARRAY_END}
```

### Recommended SLURM Parameters by Scale

| Workload Size | Array Config | Time Limit | Memory | Notes |
|---------------|--------------|------------|--------|-------|
| < 100 | `--array=0-99` | 2:00:00 | 4G | No batching needed |
| 100-10,000 | `--array=0-99%50` | 2:00:00 | 4G | Limit concurrent to 50 |
| 10,000-100,000 | `--array=0-999%100` | 4:00:00 | 4G | Use batching |
| > 100,000 | `--array=0-9999%200` | 8:00:00 | 8G | Large batch, high concurrency |

---

## Generated SLURM Files

`01_create_workload.py` automatically generates SLURM job scripts.

(source: `01_create_workload.py::create_AL_experiment_slurm_files`, lines 321-348)

### 02_slurm.slurm

Main experiment execution script:

```bash
#!/bin/bash
#SBATCH --time={{SLURM_TIME_LIMIT}}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={{ SLURM_NR_THREADS }}
#SBATCH --mem-per-cpu={{ SLURM_MEMORY }}M
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --array {{ START }}-{{ END }}

# Batching loop
i=$(( {{ SLURM_OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ SLURM_ITERATIONS_PER_BATCH }} ))
end=$(($i+{{ SLURM_ITERATIONS_PER_BATCH }}))
for ((j = $i ; j < $end ; j++)); do
    timeout {{timeout_duration}} python 02_run_experiment.py \
        --EXP_TITLE {{EXP_TITLE}} \
        --RUNNING_ENVIRONMENT hpc \
        --WORKER_INDEX $j
done
```

(source: `resources/slurm_templates/slurm_parallel.sh`)

### Key SLURM Configuration Options

| Option | Description | Default | Source |
|--------|-------------|---------|--------|
| `SLURM_TIME_LIMIT` | Wall time limit | "1:59:59" | `misc/config.py::Config.SLURM_TIME_LIMIT` |
| `SLURM_NR_THREADS` | CPUs per task | 1 | `misc/config.py::Config.SLURM_NR_THREADS` |
| `SLURM_MEMORY` | Memory per CPU (MB) | 2210 | `misc/config.py::Config.SLURM_MEMORY` |
| `SLURM_ITERATIONS_PER_BATCH` | Experiments per array task | 100 | `misc/config.py::Config.SLURM_ITERATIONS_PER_BATCH` |
| `SLURM_OFFSET` | Starting index offset | 0 | `misc/config.py::Config.SLURM_OFFSET` |

### Per-Experiment Timeout

Each experiment has an additional timeout beyond SLURM wall time:

(source: `resources/slurm_templates/slurm_parallel.sh`, line 24)

```bash
timeout_duration = config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT * (config.EXP_GRID_NUM_QUERIES[0] + 1)
```

Where `EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT` defaults to 300 seconds (5 minutes).

(source: `misc/config.py::Config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT`)

---

## Resume and Recovery

### Completion Tracking Files

OGAL tracks experiment progress via three CSV files:

(source: `misc/config.py::Config`, lines 123-125)

| File | Purpose | Updated By |
|------|---------|------------|
| `05_done_workload.csv` | Successfully completed experiments | `framework_runners/base_runner.py::AL_Experiment.run_experiment` (line 224-229) |
| `05_failed_workloads.csv` | Failed experiments with error type | `framework_runners/base_runner.py::AL_Experiment.run_experiment` (line 211-217) |
| `05_started_oom_workloads.csv` | Experiments started but presumed OOM | `framework_runners/base_runner.py::AL_Experiment.run_experiment` (line 120-126) |

### How Completion Is Detected

(source: `framework_runners/base_runner.py::AL_Experiment.run_experiment`, lines 218-229)

```python
if not error_was_being_raised:
    if not early_stopped_due_to_runtime_limit:
        for metric in self.metrics:
            metric.save_metrics(self)
    
    # Mark as complete
    with open(self.config.OVERALL_DONE_WORKLOAD_PATH, "a") as f:
        w = csv.DictWriter(f, fieldnames=self.config._original_workload.keys())
        if self.config.OVERALL_DONE_WORKLOAD_PATH.stat().st_size == 0:
            w.writeheader()
        w.writerow(self.config._original_workload)
```

### Resume Behavior on Workload Recreation

When `01_create_workload.py` detects existing tracking files, it automatically excludes completed/failed experiments:

(source: `01_create_workload.py::create_workload`, lines 103-223)

```python
if os.path.isfile(config.OVERALL_DONE_WORKLOAD_PATH):
    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
    open_workload_df = pd.read_csv(config.WORKLOAD_FILE_PATH)
    failed_workload_df = pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)
    
    # Remove completed experiments
    open_workload_df = _remove_right_from_left_workload(open_workload_df, done_workload_df)
    
    # Remove OOM experiments
    oom_workload_df = pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)
    open_workload_df = _remove_right_from_left_workload(open_workload_df, oom_workload_df)
```

### Resuming After Partial Run

```bash
# 1. Re-run workload creation (automatically excludes completed)
python 01_create_workload.py --EXP_TITLE your_experiment

# 2. Check remaining workload size
wc -l OUTPUT_PATH/your_experiment/01_workload.csv

# 3. Submit new jobs for remaining experiments
sbatch OUTPUT_PATH/your_experiment/02_slurm.slurm
```

### Rerunning Failed Experiments

Set `RERUN_FAILED_WORKLOADS: true` to include failed experiments in the new workload:

(source: `01_create_workload.py`, lines 186-210)

---

## Recommended Job Array Setup

### Small Experiment (< 1000 runs)

```bash
#SBATCH --array=0-999
#SBATCH --time=2:00:00
#SBATCH --mem=4G
```

### Medium Experiment (1000-100,000 runs)

```bash
#SBATCH --array=0-999%100    # Limit concurrent jobs to 100
#SBATCH --time=4:00:00
#SBATCH --mem=4G
```

### Large Experiment (> 100,000 runs)

```bash
#SBATCH --array=0-9999%200   # 10,000 array tasks, max 200 concurrent
#SBATCH --time=8:00:00
#SBATCH --mem=8G
```

---

## Storage and Filesystem Requirements

### Shared Filesystem Requirements

- `OUTPUT_PATH` **must** be on shared storage accessible from all compute nodes
- Datasets should be on fast storage (SSD/NVMe preferred for large datasets)
- Consider staging datasets to node-local `$TMPDIR` for I/O-intensive strategies

### Estimated Storage Per Experiment

| File Type | Size Per Experiment | Notes |
|-----------|---------------------|-------|
| Per-cycle metrics | ~1-10 KB | 7 metric files |
| Timing metrics | ~1-5 KB | Per-cycle timing |
| Predictions (Parquet) | ~10-500 KB | Optional, can be large |
| Tracking CSVs | ~100 bytes | Appended per experiment |

### Failure Recovery Tips

1. **Disk quota exceeded**: Monitor `OUTPUT_PATH` usage; compress old results with `02c_gzip_results.sh.slurm`
2. **NFS timeout**: Increase SLURM `--time` and retry; use local scratch for intermediate files
3. **Corrupted CSV files**: Run `scripts/find_broken_file.py` to identify issues

---

## HPC Troubleshooting Checklist

### Missing Dataset Paths

**Symptom:** `FileNotFoundError` for dataset files

**Verification:**

```bash
# On compute node
srun --pty bash
ls -la $DATASETS_PATH/Iris.csv
ls -la $DATASETS_PATH/Iris_split.csv
```

**Code pointer:** Dataset loading occurs in `datasets/__init__.py::load_dataset`

**Resolution:** Ensure `DATASETS_PATH` in `.server_access_credentials.cfg` is accessible from compute nodes (not just login node).

---

### Output Directory Permissions

**Symptom:** `PermissionError` or `OSError` when writing results

**Verification:**

```bash
srun --pty bash
mkdir -p $OUTPUT_PATH/test_write
touch $OUTPUT_PATH/test_write/test.txt && rm -rf $OUTPUT_PATH/test_write
```

**Code pointer:** Output directory creation in `misc/config.py::Config._pathes_magic` (line 308)

**Resolution:** Ensure write permissions on `OUTPUT_PATH` from compute nodes.

---

### Mismatched Python Environment (Login vs Compute Node)

**Symptom:** `ModuleNotFoundError` or version mismatches on compute nodes

**Verification:**

```bash
# Compare environments
# On login node:
which python
python --version
pip list | grep pandas

# On compute node:
srun --pty bash
which python
python --version
pip list | grep pandas
```

**Code pointer:** Python path set in `resources/slurm_templates/slurm_parallel.sh` via `HPC_PYTHON_PATH`

**Resolution:** 
1. Use absolute path to conda environment Python in `HPC_PYTHON_PATH`
2. Ensure environment is accessible from compute nodes (shared filesystem)
3. Use `module load` commands in SLURM script if required

---

### Worker Collisions / Duplicated Indices

**Symptom:** Duplicate entries in `05_done_workload.csv` or metric files

**Verification:**

```bash
# Check for duplicates in done workload
cut -d',' -f1 OUTPUT_PATH/your_experiment/05_done_workload.csv | sort | uniq -d
```

**Code pointer:** Worker index assignment in `misc/config.py::Config.load_workload` (line 393)

**Resolution:**
1. Ensure `SLURM_ARRAY_TASK_ID` is unique per task
2. Check that batching calculation doesn't overlap indices
3. Run `scripts/remove_duplicated_exp_ids.py` to clean up

---

### Partial Outputs / Corrupted Run Files

**Symptom:** Experiments marked done but metric files incomplete or missing

**Verification:**

```bash
# Check for experiments in done but missing metric files
python -m scripts.find_missing_exp_ids_in_metric_files --EXP_TITLE your_experiment

# Find broken CSV files
python -m scripts.find_broken_file --EXP_TITLE your_experiment
```

**Code pointers:**

- Metric saving: `framework_runners/base_runner.py::AL_Experiment.run_experiment` (lines 220-221)
- Missing ID detection: `scripts/find_missing_exp_ids_in_metric_files.py`
- Broken file detection: `scripts/find_broken_file.py`

**Resolution:**
1. Identify corrupted files with utility scripts
2. Remove affected experiment IDs from `05_done_workload.csv`
3. Recreate workload and rerun

---

### OOM (Out of Memory) Errors

**Symptom:** Jobs killed by SLURM with `OUT_OF_MEMORY` state

**Verification:**

```bash
# Check job states
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS

# Check OOM tracking file
cat OUTPUT_PATH/your_experiment/05_started_oom_workloads.csv | head
```

**Code pointer:** OOM tracking in `framework_runners/base_runner.py::AL_Experiment.run_experiment` (lines 120-126)

**Resolution:**
1. Increase `SLURM_MEMORY` in config or SLURM script
2. Exclude memory-intensive datasets/strategies
3. OOM experiments are automatically excluded from future workloads

---

### Runtime Timeout Exceeded

**Symptom:** Experiments not completing, no error in logs

**Verification:**

```bash
# Check for experiments that started but didn't finish
# (in OOM file but not in done or failed)
comm -23 <(sort 05_started_oom_workloads.csv) <(sort 05_done_workload.csv) | head
```

**Code pointer:** Runtime limit check in `framework_runners/base_runner.py::AL_Experiment.run_experiment` (lines 193-201)

```python
with stopit.ThreadingTimeout(self.config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT) as to_ctx_mgr:
    self.al_cycle(iteration_counter=iteration)
if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
    early_stopped_due_to_runtime_limit = True
```

**Resolution:**
1. Increase `EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT` (default: 300s)
2. Or accept early-stopped experiments (they are marked as complete but with partial results)

---

## Workflow Example

### Initial Setup

```bash
# On HPC login node
cd /data/workspace/al_olympics/code

# Create conda environment
conda create --prefix /data/workspace/al_olympics/conda-env --file conda-linux-64.lock
conda activate /data/workspace/al_olympics/conda-env
poetry install
```

### Running an Experiment

```bash
# 1. Create workload (on login node)
python 01_create_workload.py --EXP_TITLE full_experiment

# 2. Check workload size
wc -l OUTPUT_PATH/full_experiment/01_workload.csv

# 3. Submit SLURM job array
sbatch OUTPUT_PATH/full_experiment/02_slurm.slurm

# 4. Monitor progress
watch -n 60 'wc -l OUTPUT_PATH/full_experiment/05_done_workload.csv'
```

### Chain Jobs for Long Experiments

For experiments that exceed wall time limits:

```bash
# Submit initial job
JOB1=$(sbatch --parsable OUTPUT_PATH/full_experiment/02_slurm.slurm)

# Chain subsequent jobs with dependency
sbatch --dependency=afterany:$JOB1 OUTPUT_PATH/full_experiment/02_slurm.slurm
```

Alternatively, use the generated chain job script:

```bash
./OUTPUT_PATH/full_experiment/02b_chain_job.sh
```

(source: `01_create_workload.py::create_AL_experiment_slurm_files`, line 336)

---

## Performance Tips

### Dataset Placement

- Copy datasets to node-local SSD if available (`$TMPDIR`)
- Use parallel filesystem (Lustre, GPFS) for shared access
- Consider staging for I/O-intensive strategies

### Batch Size Tuning

- Larger `SLURM_ITERATIONS_PER_BATCH` reduces job scheduling overhead
- But increases time-to-first-result
- Balance based on expected experiment duration

### Prioritizing Fast Strategies

Some AL strategies are much faster than others. Consider:

1. Running fast strategies first to get preliminary results
2. Using `SEPARATE_HPC_LOCAL_WORKLOAD: true` to split workloads

(source: `01_create_workload.py`, lines 282-293)

### Resource Estimation

```python
# Estimate total compute time
workload_size = 1_000_000  # experiments
avg_time_per_experiment = 60  # seconds
concurrent_jobs = 200

total_hours = (workload_size * avg_time_per_experiment) / (concurrent_jobs * 3600)
print(f"Estimated wall time: {total_hours:.1f} hours")
```

---

## Additional HPC Configuration

| Option | Description | Default | Source |
|--------|-------------|---------|--------|
| `RUNNING_ENVIRONMENT` | Set to "hpc" on cluster | "local" | `misc/config.py::Config.RUNNING_ENVIRONMENT` |
| `BASH_PARALLEL_RUNNERS` | Parallel workers for local execution | 10 | `misc/config.py::Config.BASH_PARALLEL_RUNNERS` |
| `SEPARATE_HPC_LOCAL_WORKLOAD` | Split strategies unsuitable for HPC | false | `misc/config.py::Config.SEPARATE_HPC_LOCAL_WORKLOAD` |

### Example HPC Invocation

```bash
python 02_run_experiment.py \
    --EXP_TITLE my_experiment \
    --RUNNING_ENVIRONMENT hpc \
    --WORKER_INDEX $SLURM_ARRAY_TASK_ID
```
