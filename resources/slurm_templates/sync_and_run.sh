#!/bin/bash
rsync -avz -P {{ LOCAL_CODE_PATH }}/ {{ HPC_SSH_LOGIN }}:{{ HPC_CODE_PATH }} --exclude '.git/' --exclude '.mypy_cache/' --exclude '*.pyc' --exclude '__pycache__'
rsync -avz -P {{ LOCAL_OUTPUT_PATH }}/ {{ HPC_SSH_LOGIN }}:{{ HPC_OUTPUT_PATH }}
ssh {{ HPC_SSH_LOGIN }} << EOF
    cd {{ HPC_CODE_PATH }}
    export LC_ALL=en_US.utf-8
    export LANG=en_US.utf-8
    sbatch {{ HPC_OUTPUT_PATH}}/{{ EXP_TITLE }}/02_slurm.slurm
EOF
#    module load Python/3.8.6;