#!/bin/bash
cd {{ HPC_WS_PATH }}/code
export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
module load Python/3.8.6

{%if WITH_TRAINING_DATA_GENERATION %}create_synthetic_training_data_id=$(sbatch --parsable {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/01_create_synthetic_training_data.slurm){% else %}create_synthetic_training_data_id=1{% endif %}
{%if WITH_HYPER_SEARCH %}hyper_search_id=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/02_hyper_search.slurm){% endif %}
{%if WITH_TRAINING %}train_imital_id=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id{%if WITH_HYPER_SEARCH %}:$hyper_search_id{% endif %} {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/03_train_imital.slurm){% else %}train_imital_id=1{% endif %}
alipy_eva=$(sbatch --parsable --dependency=afterok:$create_synthetic_training_data_id:$train_imital_id {{HPC_WS_PATH}}/code/{{EXPERIMENT_LAUNCH_SCRIPTS}}/05_alipy_eva.slurm)
exit 0