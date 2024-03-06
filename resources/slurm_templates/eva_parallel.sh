#!/bin/bash
#SBATCH --time={{SLURM_TIME_LIMIT}}   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={{ SLURM_NR_THREADS }}
#SBATCH -A {{ HPC_SLURM_PROJECT }}
#SBATCH --output {{HPC_OUTPUT_PATH}}/{{EXP_TITLE}}/{{EVA_SCRIPT_WORKLOAD_DIR}}/{{EVA_NAME}}/02_%a_out.txt
#SBATCH --error {{HPC_OUTPUT_PATH}}/{{EXP_TITLE}}/{{EVA_SCRIPT_WORKLOAD_DIR}}/{{EVA_NAME}}/02_%a_error.txt
#SBATCH --array {{ START }}-{{ END-1 }}


#module load Python/3.8.6

export PYTHONDONTWRITEBYTECODE=1

cd {{HPC_WS_PATH}}code

i=$(( {{ SLURM_OFFSET }} + $SLURM_ARRAY_TASK_ID * {{ SLURM_ITERATIONS_PER_BATCH }} ))
end=$(($i+{{ SLURM_ITERATIONS_PER_BATCH }}))
for ((j = $i ; j < $end ; j++)); do
    {% if script_type == "script" %}MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} -m {{SCRIPTS_PATH}}.{{PYTHON_FILE}} --EXP_TITLE {{EXP_TITLE}} --RUNNING_ENVIRONMENT hpc --WORKER_INDEX $j --EVA_MODE slurm {{CLI_ARGS}}{% endif %}
    {% if script_type == "dataset_categorization" %}MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} {{PYTHON_FILE}}.py --EXP_TITLE {{EXP_TITLE}} --RUNNING_ENVIRONMENT hpc --WORKER_INDEX $j --EVA_MODE slurm {{CLI_ARGS}}{% endif %}
    {% if script_type == "metrics" %}MPLCONFIGPATH={{HPC_WS_PATH}}cache {{HPC_PYTHON_PATH}} {{PYTHON_FILE}}.py --EXP_TITLE {{EXP_TITLE}} --RUNNING_ENVIRONMENT hpc --WORKER_INDEX $j --EVA_MODE slurm {{CLI_ARGS}}{% endif %}

done
exit 0
