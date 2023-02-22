#!/bin/sh

python -m scripts.fix_duplicate_header_columns --EXP_TITLE full_exp 
python -m scripts.fix_early_stopping_dict_keys_too_small_error --EXP_TITLE full_exp 
python -m scripts.fix_macro_f1_score_duplicates --EXP_TITLE full_exp 
python -m scripts.fix_apply_runtime_limit_post_mortem --EXP_TITLE full_exp

chmod u+x /home/mg/exp_results/full_exp/02c_gzip_results.sh.slurm

cd /home/mg/exp_results/full_exp
bash /home/mg/exp_results/full_exp/02c_gzip_results.sh.slurm

cd /home/jg/al_olympics/code
python 04_calculate_all_compound_metrics.py --EXP_TITLE full_exp --COMPUTED_METRICS DISTANCE_METRICS MISMATCH_TRAIN_TEST CLASS_DISTRIBUTIONS METRIC_DROP HARDEST_SAMPLES QUERIED_FROM_OPTIMAL    
