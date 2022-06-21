#!/bin/bash
#SBATCH --time=100:59:59  # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1875M   # memory per CPU core
#SBATCH -A "p_ml_il"
#SBATCH --output /scratch/ws/1/s5968580-al_olympics//slurm_pipenv__out.txt
#SBATCH --error /scratch/ws/1/s5968580-al_olympics//slurm_pipenv__error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#module load Python/3.8.6
MPLCONFIGPATH=/scratch/ws/1/s5968580-al_olympics/cache /home/s5968580/.local/share/virtualenvs/code-SIvipOmJ/bin/python -m pipenv install
exit 0
