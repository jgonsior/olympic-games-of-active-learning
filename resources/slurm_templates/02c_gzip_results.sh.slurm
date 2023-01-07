#!/bin/bash
shopt -s globstar

for csv_file in {{HPC_WS_PATH}}exp_results/{{EXP_TITLE}}/**/*.csv; do
    if [[ "$csv_file" == */01_workload.csv ]]
    then
        continue
    fi
    if [[ "$csv_file" == */05_done_workload.csv ]]
    then
        continue
    fi
    echo $csv_file
    xz -T0 -0 $csv_file

done