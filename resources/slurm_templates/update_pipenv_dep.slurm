#!/bin/bash
#SBATCH --time=100:59:59  # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1875M   # memory per CPU core
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --output {{HPC_OUTPUT_PATH}}/slurm_pipenv__out.txt
#SBATCH --error {{HPC_OUTPUT_PATH}}/slurm_pipenv__error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE








MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} -m pipenv update
MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} -m pip uninstall libact -y
LIBACT_BUILD_VARIANCE_REDUCTION=0 MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} -m pip install git+https://github.com/jgonsior/libact.git

exit 0
