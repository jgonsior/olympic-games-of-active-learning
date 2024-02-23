#!/bin/sh
python -m scripts.reduce_to_dense --EXP_TITLE full_exp_jan

rsync -avz -P /home/thiele/exp_results/full_exp_jan/ /home/thiele/exp_results/bkp_02_full_exp_jan


python -m eva_scripts.basic_metrics_correlation --EXP_TITLE full_exp_jan


python -m eva_scripts.calculate_dataset_dependend_random_ramp_slope --EXP_TITLE full_exp_jan

python 03_calculate_dataset_categorizations.py --EXP_TITLE full_exp_jan --SAMPLES_CATEGORIZER _ALL
