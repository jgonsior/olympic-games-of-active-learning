#!/bin/bash

for i in $(seq 0 100);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.9 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.9 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.9 --EVA_MODE reduce --WORKER_INDEX $i
done

mv /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction_090


for i in $(seq 0 100);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.95 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.95 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.95 --EVA_MODE reduce --WORKER_INDEX $i
done

mv /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction_095

for i in $(seq 0 100);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.99 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.99 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.99 --EVA_MODE reduce --WORKER_INDEX $i
done

mv /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction_099

for i in $(seq 0 100);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.8 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.8 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.8 --EVA_MODE reduce --WORKER_INDEX $i
done

mv /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction_080

for i in $(seq 0 100);
do
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.7 --EVA_MODE create --WORKER_INDEX $i
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.7 --EVA_MODE local
    python -m eva_scripts.workload_reduction --EXP_TITLE full_exp_jan --EVA_WORKLOAD_REDUCTION_THRESHOLD 0.7 --EVA_MODE reduce --WORKER_INDEX $i
done

mv /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction /home/thiele/exp_results/full_exp_jan/workloads/workload_reduction_070