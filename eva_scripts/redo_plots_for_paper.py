from itertools import combinations
import multiprocessing
import subprocess
import sys
from typing import Literal
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from pathlib import Path

from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)


parquet_files = [
    "leaderboard_single_hyperparameter_influence/auc_metric_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/standard_metric_kendall.parquet",
    "basic_metrics/Standard Metrics.parquet",
    "AUC/auc_macro_f1-score.parquet",
    "runtime/query_selection_time.parquet",
    "single_learning_curve/weighted_f1-score.parquet",
    "single_learning_curve/single_exemplary_learning_curve.parquet",
    "error",
    "single_hyperparameter/EXP_BATCH_SIZE/single_hyper_EXP_BATCH_SIZE_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_DATASET/single_hyper_EXP_DATASET_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_hyper_EXP_LEARNER_MODEL_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_START_POINT/single_hyper_EXP_START_POINT_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_STRATEGY/single_hyper_EXP_STRATEGY_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_TRAIN_TEST_BUCKET_SIZE/single_hyper_EXP_TRAIN_TEST_BUCKET_SIZE_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_BATCH_SIZE/single_indice_EXP_BATCH_SIZE_full_auc__selected_indices_jaccard.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_indice_EXP_LEARNER_MODEL_full_auc__selected_indices_jaccard.parquet",
    "single_hyperparameter/EXP_STRATEGY/single_indice_EXP_STRATEGY_full_auc__selected_indices_jaccard.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper2_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_BATCH_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_LEARNER_MODEL_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_START_POINT_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_BATCH_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_DATASET_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_LEARNER_MODEL_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_START_POINT_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
]

for pf in parquet_files:
    print(pf)
    corrmat_df = pd.read_parquet(config.OUTPUT_PATH / f"plots/{pf}")

    match pf:
        case "basic_metrics/Standard Metrics.parquet":
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=True,
                rotation=30,
            )
        case "AUC/auc_macro_f1-score.parquet":
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=True,
                rotation=30,
            )
        case "leaderboard_single_hyperparameter_influence/auc_metric_kendall.parquet":
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=True,
                rotation=30,
            )
        case "leaderboard_single_hyperparameter_influence/standard_metric_kendall.parquet":
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=True,
                rotation=30,
            )
        case "runtime/query_selection_time.parquet":
            print("hui")
        case "single_learning_curve/weighted_f1-score.parquet":
            print("hui")
        case "single_learning_curve/single_exemplary_learning_curve.parquet":
            print("hui")
        case _:
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
            )
