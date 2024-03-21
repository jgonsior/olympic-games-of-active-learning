#!/bin/bash

for i in $(seq 0 1000);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_MODE reduce --WORKER_INDEX $i
done
