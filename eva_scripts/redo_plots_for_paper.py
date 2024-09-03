from itertools import combinations
import multiprocessing
import subprocess
import sys
from typing import Literal
from matplotlib import pyplot as plt, ticker
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from pathlib import Path

from misc.helpers import (
    _calculate_fig_size,
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)


parquet_files = [
    "leaderboard_invariances/leaderboard_types_kendall.parquet",
    "eee",
    "leaderboard_single_hyperparameter_influence/EXP_LEARNER_MODEL_kendall.parquet",
    # "errr",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_START_POINT_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper2_kendall.parquet",
    # "error",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_BATCH_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_LEARNER_MODEL_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_START_POINT_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_hyper_EXP_LEARNER_MODEL_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_indice_EXP_LEARNER_MODEL_full_auc__selected_indices_jaccard.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_BATCH_SIZE_kendall.parquet",
    "single_hyperparameter/EXP_BATCH_SIZE/single_hyper_EXP_BATCH_SIZE_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_BATCH_SIZE/single_indice_EXP_BATCH_SIZE_full_auc__selected_indices_jaccard.parquet",
    # "error",
    "leaderboard_single_hyperparameter_influence/EXP_DATASET_kendall.parquet",
    "AUC/auc_weighted_f1-score.parquet",
    "basic_metrics/Standard Metrics.parquet",
    "leaderboard_single_hyperparameter_influence/auc_metric_kendall.parquet",
    "single_learning_curve/single_exemplary_learning_curve.parquet",
    "leaderboard_single_hyperparameter_influence/standard_metric_kendall.parquet",
    "runtime/query_selection_time.parquet",
    "single_hyperparameter/EXP_START_POINT/single_hyper_EXP_START_POINT_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_STRATEGY/single_hyper_EXP_STRATEGY_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_TRAIN_TEST_BUCKET_SIZE/single_hyper_EXP_TRAIN_TEST_BUCKET_SIZE_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_STRATEGY/single_indice_EXP_STRATEGY_full_auc__selected_indices_jaccard.parquet",
    "single_hyperparameter/EXP_DATASET/single_hyper_EXP_DATASET_full_auc_weighted_f1-score.parquet",
    "single_learning_curve/weighted_f1-score.parquet",
    # "error",
    "eee",
    # "error",
]

for pf in parquet_files:
    print(pf)
    corrmat_df = pd.read_parquet(config.OUTPUT_PATH / f"plots/{pf}")

    match pf:
        case "single_learning_curve/weighted_f1-score.parquet":
            set_seaborn_style(font_size=7)

            plt.figure(figsize=_calculate_fig_size(3.57))
            ax = sns.lineplot(corrmat_df, x="ix", y="metric_value", hue="EXP_STRATEGY")
            ax.set(title="", xlabel="AL Cycle", ylabel="Class Weighted F1-Score")
            plt.legend([], [], frameon=False)

            plt.savefig(
                config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        case "single_learning_curve/single_exemplary_learning_curve.parquet":
            set_seaborn_style(font_size=7)
            # plt.figure(figsize=set_matplotlib_size(fraction=10))

            # calculate fraction based on length of keys
            plt.figure(figsize=_calculate_fig_size(3.57))
            ax = sns.lineplot(corrmat_df)
            ax.set(title="", xlabel="AL Cycle", ylabel="ML Performance Metric")

            ax.xaxis.set_major_locator(
                ticker.FixedLocator([rrr for rrr in range(0, 10)])
            )
            # ax.set_title(f"Learning Curve: {standard_metric}")

            plt.savefig(
                config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
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
            set_seaborn_style(font_size=7)
            plt.figure(figsize=_calculate_fig_size(3.57, heigh_bonus=0.9))

            ax = sns.barplot(
                data=corrmat_df, x="EXP_STRATEGY", y="mean", hue="EXP_STRATEGY"
            )
            ax.set(xlabel=None)
            ax.set_yscale("log")
            # ax.get_yaxis().get_major_formatter().labelOnlyBase = False
            # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(
                    lambda y, pos: (
                        "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(y), 0)))
                    ).format(y)
                )
            )
            ax.set_ylim(0, 260)
            ax.set(ylabel="Mean Duration of AL cycle in seconds")

            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

            xs = []
            for container in ax.containers:
                ax.bar_label(container, padding=1, fmt="%.2f", rotation=90)

                xs.append(container.datavalues[0])

            plt.savefig(
                config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        case _:

            if pf == "leaderboard_invariances/leaderboard_types_kendall.parquet":
                corrmat_df = corrmat_df.loc[
                    ~corrmat_df.index.str.endswith("sparse_average_of_same_strategy"),
                    :,
                ]
                corrmat_df = corrmat_df.loc[
                    ~corrmat_df.index.str.endswith("sparse_remove"), :
                ]
                corrmat_df = corrmat_df.loc[
                    :,
                    ~corrmat_df.columns.str.endswith("sparse_average_of_same_strategy"),
                ]
                corrmat_df = corrmat_df.loc[
                    :, ~corrmat_df.columns.str.endswith("sparse_remove")
                ]
                corrmat_df = corrmat_df.loc[
                    ~corrmat_df.columns.str.endswith("dense"), :
                ]
                corrmat_df = corrmat_df.loc[
                    :, ~corrmat_df.columns.str.endswith("dense")
                ]
            elif pf.startswith("leaderboard_single_hyperparameter_influence/min_hyper"):
                set_seaborn_style(font_size=6)
                plt.figure(figsize=_calculate_fig_size(2))

                corrmat_df["index"] = corrmat_df["index"].astype("int64")

                ax = sns.regplot(
                    data=corrmat_df,
                    x="index",
                    # x_estimator=lambda x: np.percentile(x, 95),
                    y="spearman",
                    ci=None,
                    color=".5",
                    line_kws=dict(color="darkorange"),
                    order=3,
                    marker="x",
                    scatter_kws={"s": 0.01, "rasterized": True},
                    # robust=True,
                    # logistic=True,
                    # lowess=True,
                    # lowess=True,  # order=4,
                    # errorbar=lambda x: (x.min(), x.max()),
                    #  errorbar="sd",:w
                    # errorbar="sd",
                    # errorbar="ci",
                    # sizes=(0.1, 0.1),
                    #  alpha=0.2,
                    # edgecolor="none",
                    #  hue=0.3,
                )
                # ax.set_title("test")
                #  ax.xaxis.set_major_locator(ticker.LinearLocator(20))
                # ax.xaxis.set_major_locator(ticker.AutoLocator())
                ax.set(xlabel="# hyperparameter combinations")
                ax.set(ylabel="spearman coefficient")
                ax.set(ylim=(0, 1))

                plt.savefig(
                    config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                continue

            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
            )
