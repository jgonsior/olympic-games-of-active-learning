#!/bin/bash
#SBATCH --time={{SLURM_TIME_LIMIT}}   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{ SLURM_NR_THREADS }}
#SBATCH --mem-per-cpu={{ SLURM_MEMORY }}M   # memory per CPU core
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --output {{HPC_WS_PATH}}/slurm_{{EXP_TITLE}}_{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_WS_PATH}}/slurm_{{EXP_TITLE}}_{{PYTHON_FILE}}_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#module load Python/3.8.6
tar -czf {{HPC_WS_PATH}}exp_results/{{EXP_TITLE}}.tar.gz {{HPC_WS_PATH}}exp_results/{{EXP_TITLE}}
exit 0