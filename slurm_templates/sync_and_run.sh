#!/bin/bash
rsync -avz -P {{ LOCAL_CODE_PATH }} {{ HPC_SSH_LOGIN }}:{{ HPC_WS_PATH }}/code --exclude '.git/' --exclude '.mypy_cache/'
rsync -avz -P {{ LOCAL_OUTPUT_PATH }} {{ HPC_SSH_LOGIN }}:{{ HPC_WS_PATH }}/exp_results
ssh {{ HPC_SSH_LOGIN }} << EOF
    cd {{ HPC_WS_PATH }}/exp_results
    export LC_ALL=en_US.utf-8
    export LANG=en_US.utf-8
    module load Python/3.8.6;
    sbatch {{ EXP_TITLE }}/02_slurm.slurm
EOF