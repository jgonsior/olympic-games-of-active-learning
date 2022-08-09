#!/bin/bash
cd {{ HPC_CODE_PATH }}
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

workload_run=$(sbatch --parsable {{ HPC_OUTPUT_PATH}}/{{ EXP_TITLE }}/02_slurm.slurm)
tar_file=$(sbatch --parsable --dependency=afterok:$workload_run {{ HPC_OUTPUT_PATH}}/{{ EXP_TITLE }}/03_tar.slurm)