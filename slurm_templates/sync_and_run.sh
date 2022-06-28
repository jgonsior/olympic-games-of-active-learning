#!/bin/bash
rsync -avz -P {{ LOCAL_CODE_PATH }}/ {{ HPC_SSH_LOGIN }}:{{ HPC_CODE_PATH }} --exclude '.git/' --exclude '.mypy_cache/'
rsync -avz -P {{ LOCAL_OUTPUT_PATH }}/ {{ HPC_SSH_LOGIN }}:{{ HPC_OUTPUT_PATH }}
ssh {{ HPC_SSH_LOGIN }} << EOF
    {{ HPC_OUTPUT_PATH}}/{{ EXP_TITLE }}/02b_chainjob.sh)
EOF