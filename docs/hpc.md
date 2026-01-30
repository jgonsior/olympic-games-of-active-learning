# HPC Setup Guide

This document explains how to run OGAL experiments on High-Performance Computing (HPC) clusters using SLURM job arrays.

## Overview

OGAL is designed for massive parallelization on HPC clusters. The key mechanism is:

1. **Workload partitioning**: `01_create_workload.py` generates a CSV with thousands/millions of experiment configurations
2. **Worker sharding**: Each SLURM job array task runs `02_run_experiment.py` with a different `WORKER_INDEX`
3. **Result aggregation**: Results are written to shared storage and tracked in `05_done_workload.csv`

---

## Server Access Credentials

**File:** `.server_access_credentials.cfg`

This file configures paths and credentials for HPC access.

### HPC Section

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
```

### Field Reference

| Field | Description |
|-------|-------------|
| `SSH_LOGIN` | SSH connection string for remote access |
| `WS_PATH` | Base workspace path on HPC filesystem |
| `DATASETS_PATH` | Path to datasets (should be on fast storage) |
| `OUTPUT_PATH` | Path for experiment outputs (should be on shared storage) |
| `SLURM_MAIL` | Email for job notifications |
| `SLURM_PROJECT` | SLURM account/project for resource allocation |
| `CODE_PATH` | Path to the OGAL repository on HPC |
| `PYTHON_PATH` | Path to Python interpreter in conda environment |

---

## WORKER_INDEX Mechanism

The `WORKER_INDEX` parameter selects which row from the workload CSV to execute.

### How It Works

1. `01_workload.csv` contains N rows (experiments)
2. `WORKER_INDEX=0` runs row 0
3. `WORKER_INDEX=1` runs row 1
4. ...
5. `WORKER_INDEX=N-1` runs row N-1

### SLURM Job Arrays

SLURM job arrays map `SLURM_ARRAY_TASK_ID` to `WORKER_INDEX`:

```bash
# In SLURM script
python 02_run_experiment.py --EXP_TITLE my_experiment --WORKER_INDEX $SLURM_ARRAY_TASK_ID
```

---

## Generated SLURM Files

`01_create_workload.py` automatically generates SLURM job scripts:

### 02_slurm.slurm

Main experiment execution script:

```bash
#!/bin/bash
#SBATCH --job-name=al_exp
#SBATCH --array=0-999      # Adjust based on workload size
#SBATCH --time=1:59:59
#SBATCH --mem=2210M
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=your.email@example.edu
#SBATCH --account=your_project

cd /path/to/code
source activate conda-env

python 02_run_experiment.py \
    --EXP_TITLE my_experiment \
    --RUNNING_ENVIRONMENT hpc \
    --WORKER_INDEX $SLURM_ARRAY_TASK_ID
```

### Batching Strategy

For large workloads, experiments are batched (source: `misc/config.py::Config.SLURM_ITERATIONS_PER_BATCH`):

- `SLURM_ITERATIONS_PER_BATCH`: Number of experiments per array task (default: 100)
- Array task 0 runs experiments 0-99
- Array task 1 runs experiments 100-199
- etc.

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

### Key SLURM Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--time` | Wall time limit | 2-8 hours per batch |
| `--mem` | Memory per task | 2-8 GB depending on dataset size |
| `--cpus-per-task` | CPU cores | 1 (single-threaded AL) |
| `--array` | Task indices | Match workload size / batch size |
| `%N` suffix | Concurrent limit | 100-500 to avoid overwhelming scheduler |

---

## Storage and Resume Strategy

### Shared Filesystem Requirements

- `OUTPUT_PATH` must be on shared storage accessible from all compute nodes
- Datasets should be on fast storage (SSD/NVMe preferred)
- Results are appended to CSV files (requires file locking awareness)

### Automatic Resume

OGAL tracks experiment progress automatically:

1. **`05_done_workload.csv`**: Successfully completed experiments
2. **`05_failed_workloads.csv`**: Failed experiments with error types
3. **`05_started_oom_workloads.csv`**: OOM-killed experiments

### Resuming After Failure

```bash
# 1. Re-run workload creation to identify remaining work
python 01_create_workload.py --EXP_TITLE my_experiment

# 2. Check how many experiments remain
wc -l OUTPUT_PATH/my_experiment/01_workload.csv

# 3. Submit new SLURM jobs for remaining experiments
sbatch OUTPUT_PATH/my_experiment/02_slurm.slurm
```

### Handling OOM Failures

Experiments that trigger OOM are tracked separately. To exclude them from future runs:

```python
# The workload creation automatically excludes OOM experiments
# by reading 05_started_oom_workloads.csv
```

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
# 1. Create workload
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

---

## Troubleshooting

### Job Array Tasks Failing

```bash
# Check failed task logs
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS

# Look for specific task errors
cat slurm-<JOBID>_<ARRAYID>.out
```

### OOM Errors

1. Increase `--mem` in SLURM script
2. Or exclude memory-intensive datasets/strategies
3. Check `05_started_oom_workloads.csv` for patterns

### File Lock Contention

If many tasks try to write to the same CSV simultaneously:

1. Results may be duplicated or corrupted
2. Consider post-processing to deduplicate
3. Use separate output directories per strategy if needed

### Dataset Access Issues

```bash
# Verify datasets are accessible from compute nodes
srun --pty bash
ls -la $DATASETS_PATH
```

---

## Performance Tips

### Dataset Placement

- Copy datasets to node-local SSD if available (`$TMPDIR`)
- Use parallel filesystem (Lustre, GPFS) for shared access

### Batch Size Tuning

- Larger `SLURM_ITERATIONS_PER_BATCH` reduces job scheduling overhead
- But increases time-to-first-result

### Prioritizing Fast Strategies

Some AL strategies are much faster than others. Consider:

1. Running fast strategies first to get preliminary results
2. Using `SEPARATE_HPC_LOCAL_WORKLOAD: true` to split workloads

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

## HPC-Specific Configuration

### Additional Config Options

| Option | Description | Default | Source |
|--------|-------------|---------|--------|
| `RUNNING_ENVIRONMENT` | Set to "hpc" on cluster | "local" | `misc/config.py::Config.RUNNING_ENVIRONMENT` |
| `SLURM_TIME_LIMIT` | SLURM time limit string | "1:59:59" | `misc/config.py::Config.SLURM_TIME_LIMIT` |
| `SLURM_NR_THREADS` | CPUs per task | 1 | `misc/config.py::Config.SLURM_NR_THREADS` |
| `SLURM_MEMORY` | Memory per task (MB) | 2210 | `misc/config.py::Config.SLURM_MEMORY` |
| `SLURM_OFFSET` | Starting array index | 0 | `misc/config.py::Config.SLURM_OFFSET` |
| `SLURM_ITERATIONS_PER_BATCH` | Experiments per array task | 100 | `misc/config.py::Config.SLURM_ITERATIONS_PER_BATCH` |

### Setting Environment

```bash
python 02_run_experiment.py \
    --EXP_TITLE my_experiment \
    --RUNNING_ENVIRONMENT hpc \
    --WORKER_INDEX $SLURM_ARRAY_TASK_ID
```
