#!/bin/sh

#python -m scripts.fix_duplicate_header_columns --EXP_TITLE full_exp --LOCAL_OUTPUT_PATH /home/mg/exp_results/full_exp.taurus
python -m scripts.fix_early_stopping_dict_keys_too_small_error --EXP_TITLE full_exp --LOCAL_OUTPUT_PATH /home/mg/exp_results/full_exp.taurus
python -m scripts.fix_macro_f1_score_duplicates --EXP_TITLE full_exp --LOCAL_OUTPUT_PATH /home/mg/exp_results/full_exp.taurus
python -m scripts.fix_apply_runtime_limit_post_mortem --EXP_TITLE full_exp --LOCAL_OUTPUT_PATH /home/mg/exp_results/full_exp.taurus

