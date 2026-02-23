# Run at HPC Scale

Submit thousands of experiments to a SLURM cluster.

---

## Prerequisites

- A working SLURM cluster with SSH access
- The `[HPC]` section of `.server_access_credentials.cfg` filled in (see [Configuration](configuration.md))

## 1. Generate the workload and SLURM script

```bash
python 01_create_workload.py --EXP_TITLE my_run --RUNNING_ENVIRONMENT hpc
```

This produces `01_workload.csv` (one row per experiment) and `02_slurm.slurm`.

## 2. Submit to SLURM

```bash
sbatch {OUTPUT_PATH}/my_run/02_slurm.slurm
```

## 3. Post-process

After all jobs complete, run the post-processing steps described in [Pipeline → Steps 3–6](pipeline.md#step-3-dataset-categorizations).

---

For full details on SLURM keys (`SLURM_TIME_LIMIT`, `SLURM_MEMORY`, etc.), see [Configuration](configuration.md#slurm-keys-hpc-only).

**Next:** [Pipeline](pipeline.md) · [Configuration](configuration.md) · [Use OPARA archive](use_opara_archive.md)
