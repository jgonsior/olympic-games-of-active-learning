#!/bin/bash{% if array %}{% set THREADS = 1 %}{% set MEMORY = 1875 %}{% endif %}
#SBATCH --partition=haswell,romeo
#SBATCH --time={{TIME_LIMIT}}   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{ THREADS }}
#SBATCH --mem-per-cpu={{ MEMORY }}M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output {{HPC_WS_PATH}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_WS_PATH}}/slurm_{{TITLE}}_{{PYTHON_FILE}}_error.txt
{% if array %}#SBATCH --array {{START}}-{{END}}{% endif %}

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module load Python/3.8.6

{% if array %}i=$(( {{ OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ ITERATIONS_PER_BATCH }} )){% endif %}
module load Python/3.8.6

MPLCONFIGPATH={{HPC_WS_PATH}}/cache python3 -m pipenv run python {{HPC_WS_PATH}}/code/{{PYTHON_FILE}}.py {{ CLI_ARGS }} {% if APPEND_OUTPUT_PATH %} {{ OUTPUT_PATH }}/{{ EXP_TITLE }} {% endif %}
exit 0