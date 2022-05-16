#!/bin/bash{% if array %}{% set THREADS = 1 %}{% set MEMORY = 1875 %}{% endif %}
#SBATCH --partition=haswell,romeo
#SBATCH --time={{SLURM_TIME_LIMIT}}   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{ SLURM_NR_THREADS }}
#SBATCH --mem-per-cpu={{ SLURM_MEMORY }}M   # memory per CPU core
#SBATCH --mail-user={{ HPC_SLURM_MAIL }}
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --output {{HPC_WS_PATH}}/slurm_{{EXP_TITLE}}_{{PYTHON_FILE}}_out.txt
#SBATCH --error {{HPC_WS_PATH}}/slurm_{{EXP_TITLE}}_{{PYTHON_FILE}}_error.txt
{% if array %}#SBATCH --array {{ START }}-{{ END }}{% endif %}

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

module load Python/3.8.6

{% if array %}i=$(( {{ SLURM_OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ SLURM_ITERATIONS_PER_BATCH }} )){% endif %}
#module load Python/3.8.6
end=$(($i+{{ SLURM_ITERATIONS_PER_BATCH }}))
for ((j = $i ; j < $end ; j++)); do
    MPLCONFIGPATH={{HPC_WS_PATH}}/cache python3 -m pipenv run python {{HPC_WS_PATH}}/code/{{PYTHON_FILE}} --EXP_TITLE {{EXP_TITLE}} {{ CLI_ARGS }} {% if APPEND_OUTPUT_PATH %} {{ OUTPUT_PATH }}/{{ EXP_TITLE }} {% endif %} --WORKER_INDEX $j
done
exit 0