import enum
from itertools import combinations
import multiprocessing
import subprocess
import sys
from typing import Literal
from matplotlib import pyplot as plt, ticker
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.patches as mpatches
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
    "runtime/query_selection_time.parquet",
    "eee",
    "AUC/auc_weighted_f1-score.parquet",
    "basic_metrics/Standard Metrics.parquet",
    "final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet",
    # "final_leaderboard/dataset_normalized_percentages_sparse_zero_full_auc_weighted_f1-score.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_LEARNER_MODEL_kendall.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_hyper_EXP_LEARNER_MODEL_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_LEARNER_MODEL/single_indice_EXP_LEARNER_MODEL_full_auc__selected_indices_jaccard.parquet",
    "leaderboard_single_hyperparameter_influence/EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
    "single_hyperparameter/EXP_TRAIN_TEST_BUCKET_SIZE/single_hyper_EXP_TRAIN_TEST_BUCKET_SIZE_full_auc_weighted_f1-score.parquet",
    # "eee",
    "leaderboard_single_hyperparameter_influence/EXP_START_POINT_kendall.parquet",
    "single_hyperparameter/EXP_START_POINT/single_hyper_EXP_START_POINT_full_auc_weighted_f1-score.parquet",
    "leaderboard_invariances/leaderboard_types_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/standard_metric_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/auc_metric_kendall.parquet",
    #  "framework_al_strat_correlation.parquet",
    # "eee",
    "single_hyperparameter/EXP_STRATEGY/single_hyper_EXP_STRATEGY_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_STRATEGY/single_indice_EXP_STRATEGY_full_auc__selected_indices_jaccard.parquet",
    # "errr",
    "leaderboard_single_hyperparameter_influence/EXP_BATCH_SIZE_kendall.parquet",
    "single_hyperparameter/EXP_BATCH_SIZE/single_hyper_EXP_BATCH_SIZE_full_auc_weighted_f1-score.parquet",
    "single_hyperparameter/EXP_BATCH_SIZE/single_indice_EXP_BATCH_SIZE_full_auc__selected_indices_jaccard.parquet",
    # "error",
    "leaderboard_single_hyperparameter_influence/EXP_DATASET_kendall.parquet",
    "single_learning_curve/single_exemplary_learning_curve.parquet",
    "single_hyperparameter/EXP_DATASET/single_hyper_EXP_DATASET_full_auc_weighted_f1-score.parquet",
    "single_learning_curve/weighted_f1-score.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_START_POINT_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_TRAIN_TEST_BUCKET_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper2_kendall.parquet",
    # "error",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_BATCH_SIZE_kendall.parquet",
    "leaderboard_single_hyperparameter_influence/min_hyper_reduction_EXP_LEARNER_MODEL_kendall.parquet",
    # "error",
    "eee",
    # "error",
]

for pf in parquet_files:
    print(pf)
    corrmat_df = pd.read_parquet(config.OUTPUT_PATH / f"plots/{pf}")

    match pf:
        # case "final_leaderboard/dataset_normalized_percentages_sparse_zero_full_auc_weighted_f1-score.parquet":
        case "final_leaderboard/rank_sparse_zero_full_auc_weighted_f1-score.parquet":
            set_seaborn_style(font_size=3.5)

            print(corrmat_df)
            plt.figure(figsize=_calculate_fig_size(3.57 * 2, heigh_bonus=1.35))

            def sort_index(index: np.ndarray):
                print(index)
                index = [iii.lower() if iii != "Total" else "zzz" for iii in index]
                print(index)
                return index

            corrmat_df = corrmat_df.sort_index(axis=0, key=sort_index)

            ax = sns.heatmap(
                corrmat_df,
                annot=True,
                # fmt=".1f",
                cbar_kws={"shrink": 0.3},
                # vmin=0,
                # vmax=100,
                # cmap=cmap,
                # square=True,
            )
            # ax = sns.heatmap(data_df, annot=True, fmt=".2%", vmin=0, vmax=1)
            ax.tick_params(left=False, bottom=False, pad=-4)
            # ax.set(title="", xlabel="AL cycle", ylabel="class weighted F1-score")
            plt.legend([], [], frameon=False)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
            )

            plt.savefig(
                config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        case "single_learning_curve/weighted_f1-score.parquet":
            set_seaborn_style(font_size=7)

            plt.figure(figsize=_calculate_fig_size(3.57))
            ax = sns.lineplot(corrmat_df, x="ix", y="metric_value", hue="EXP_STRATEGY")
            ax.set(title="", xlabel="AL cycle", ylabel="class weighted F1-score")
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
                total=False,
                rotation=30,
            )
        case "AUC/auc_macro_f1-score.parquet":
            set_seaborn_style(font_size=5)
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=True,
                rotation=30,
            )
        case "leaderboard_single_hyperparameter_influence/auc_metric_kendall.parquet":
            set_seaborn_style(font_size=5)
            corrmat_df = corrmat_df.loc[
                ~corrmat_df.index.str.contains("gold standard"),
                :,
            ]
            print(corrmat_df)

            corrmat_df = corrmat_df.loc[
                :,
                ~corrmat_df.columns.str.contains("gold standard"),
            ]
            print(corrmat_df)
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=False,
                rotation=30,
            )
        case "leaderboard_single_hyperparameter_influence/standard_metric_kendall.parquet":
            corrmat_df = corrmat_df.loc[
                ~corrmat_df.index.str.contains("gold standard"),
                :,
            ]
            corrmat_df = corrmat_df.loc[
                :,
                ~corrmat_df.columns.str.contains("gold standard"),
            ]
            print(corrmat_df)
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=False,
                rotation=30,
            )
        case "runtime/query_selection_time.parquet":
            set_seaborn_style(font_size=6)
            plt.figure(figsize=_calculate_fig_size(3.57 * 2, heigh_bonus=0.4))

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

            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor"
            )

            xs = []
            for container in ax.containers:
                ax.bar_label(container, padding=1, fmt="%.2f", rotation=90)

                xs.append(container.datavalues[0])

            # Define some hatches
            hatches = {
                "ALI": "//",
                "SM": "oo",
                "LIB": "\\\\",
                "SKA": "..",
                "OG": "++",
            }

            for x_tick_label, (i, thisbar) in zip(
                ax.get_xticklabels(), enumerate(ax.patches)
            ):
                print(thisbar)
                print(x_tick_label.get_text())
                # Set a different hatch for each bar
                if "(" in x_tick_label.get_text():
                    framework_name = x_tick_label.get_text().split("(")[1][:-1]
                else:
                    framework_name = "OG"
                thisbar.set_hatch(hatches[framework_name])

            legend_hatches = []
            for k, v in hatches.items():
                legend_hatches.append(
                    mpatches.Patch(facecolor="#222222", alpha=0.6, hatch=v, label=k)
                )

            ax.legend(
                handles=legend_hatches,
                loc=2,
                handleheight=4,
                handlelength=1.4,
                ncol=len(legend_hatches),
                columnspacing=0.8,
            )

            plt.savefig(
                config.OUTPUT_PATH / f"plots/{pf.split('.parquet')[0]}.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
        case "leaderboard_invariances/leaderboard_types_kendall.parquet":
            corrmat_df = corrmat_df.loc[
                ~corrmat_df.index.str.contains("sparse_average_of_same_strategy"),
                :,
            ]
            corrmat_df = corrmat_df.loc[
                ~corrmat_df.index.str.contains("sparse_remove"), :
            ]
            corrmat_df = corrmat_df.loc[
                :,
                ~corrmat_df.columns.str.contains("sparse_average_of_same_strategy"),
            ]
            corrmat_df = corrmat_df.loc[
                :, ~corrmat_df.columns.str.contains("sparse_remove")
            ]
            # corrmat_df = corrmat_df.loc[~corrmat_df.columns.str.contains("dense"), :]
            # corrmat_df = corrmat_df.loc[:, ~corrmat_df.columns.str.contains("dense")]
            # print(corrmat_df.keys())
            # exit(-1)
            save_correlation_plot(
                data=corrmat_df.to_numpy(),
                title=pf.split(".parquet")[0],
                keys=corrmat_df.columns,
                config=config,
                total=False,
            )
        case _:
            if pf.startswith("leaderboard_single_hyperparameter_influence/min_hyper"):
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
