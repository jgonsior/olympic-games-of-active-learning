#!/bin/bash

for i in $(seq 0 1000);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE test2 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE test2 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE test2 --EVA_MODE reduce --WORKER_INDEX $i
done