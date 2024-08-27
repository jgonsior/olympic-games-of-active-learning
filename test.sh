#!/bin/sh
#python -m scripts.reduce_to_dense --EXP_TITLE full_exp_jan
#python -m scripts.merge_two_workloads --EXP_TITLE full_exp_jan
#rsync -avz -P /home/thiele/exp_results/bkp_01_full_exp_jan/ s5968580@login2.barnard.hpc.tu-dresden.de:/data/horse/ws/s5968580-al_olympics_jan.bak/exp_results/bkp01
#rsync -avz -P /home/thiele/exp_results/bkp_02_full_exp_jan/ s5968580@login2.barnard.hpc.tu-dresden.de:/data/horse/ws/s5968580-al_olympics_jan.bak/exp_results/bkp02
#rsync -avz -P /home/thiele/exp_results/bkp_03_full_exp_jan/ s5968580@login2.barnard.hpc.tu-dresden.de:/data/horse/ws/s5968580-al_olympics_jan.bak/exp_results/bkp03

#python -m scripts.find_missing_exp_ids_in_metric_files --EXP_TITLE full_exp_jan



#python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan

#rsync -avz -P /home/thiele/exp_results/full_exp_jan/ /home/thiele/exp_results/bkp_04_full_exp_jan
#rsync -avz -P /home/thiele/exp_results/bkp_04_full_exp_jan/ s5968580@login2.barnard.hpc.tu-dresden.de:/data/horse/ws/s5968580-al_olympics_jan.bak/exp_results/bkp04

#python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE full_exp_jan

#python 03_calculate_dataset_categorizations.py --EXP_TITLE full_exp_jan --SAMPLES_CATEGORIZER _ALL
#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS adv_start_point_scenario

#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS adv_min
#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE create --SCENARIOS min_hyper
#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS min_hyper
#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS dataset_scenario
#python -m eva_scripts.leaderboard_scenarios --EXP_TITLE full_exp_jan --EVA_MODE local --SCENARIOS start_point_scenario
python -m eva_scripts.runtime --EXP_TITLE full_exp_jan
python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.single_hyperparameter_evaluation_indices --EXP_TITLE full_exp_jan    
python -m eva_scripts.auc_metric_correlation --EXP_TITLE full_exp_jan
python -m eva_scripts.leaderboard_single_hyperparameter_influence_analyze --EXP_TITLE full_exp_jan
