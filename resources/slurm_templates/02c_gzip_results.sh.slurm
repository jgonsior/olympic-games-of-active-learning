#!/bin/bash
#SBATCH --time=100:59:59   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1700M   # memory per CPU core
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --output {{HPC_OUTPUT_PATH}}/{{EXP_TITLE}}{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_OUTPUT_PATH}}/{{EXP_TITLE}}{{PYTHON_FILE}}_error.txt

shopt -s globstar

for csv_file in ./**/*.csv; do
    if [[ "$csv_file" == */01_workload.csv ]]
    then
        continue
    fi
    if [[ "$csv_file" == */05_done_workload.csv ]]
    then
        continue
    fi
    echo $csv_file
    xz -T0 -0 $csv_file

donesrun --pty --ntasks=1 --cpus-per-task=4 --time=1:00:00 --mem-per-cpu=1700 bash -l