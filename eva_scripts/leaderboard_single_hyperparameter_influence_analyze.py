import csv
import multiprocessing
import subprocess
import sys
from typing import Dict
from matplotlib import pyplot as plt, transforms
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.style as mplstyle
from sklearn.preprocessing import RobustScaler
import matplotlib as mpl
import scipy
from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size, set_seaborn_style
from resources.data_types import AL_STRATEGY, LEARNER_MODEL
import seaborn as sns
from pprint import pprint
import ast

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

hyperparameters_to_evaluate = [
    # "random_seed_scenarios",
    # "dataset_scenarios",
    "standard_metric",
    # "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_START_POINT",
    "auc_metric",
]


rankings_df: pd.DataFrame = pd.DataFrame()
for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
    ranking_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
    )
    ranking_df = pd.read_csv(ranking_path, index_col=0).T

    keys = {
        kkk: kkk.removeprefix(f"{hyperparameter_to_evaluate}: ")
        for kkk in ranking_df.columns
    }

    ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate == "EXP_LEARNER_MODEL":
        keys = {kkk: LEARNER_MODEL(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_BATCH_SIZE":
        keys = {kkk: int(kkk) for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_DATASET":
        keys = {kkk: DATASET(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate in ["random_seed_scenarios", "dataset_scenarios"]:
        print(ranking_df)
        custom_dict = {
            v: k
            for k, v in enumerate(
                sorted(
                    ranking_df.columns, key=lambda kkk: int(ast.literal_eval(kkk)[1])
                )
            )
        }
        ranking_df = ranking_df.sort_index(axis=0)
        ranking_df = ranking_df.sort_index(key=lambda x: x.map(custom_dict), axis=1)
    else:
        ranking_df = ranking_df.sort_index(axis=0)
        ranking_df = ranking_df.sort_index(axis=1)

    keys = {kkk: f"{hyperparameter_to_evaluate}: {kkk}" for kkk in ranking_df.columns}

    ranking_df.rename(columns=keys, inplace=True)

    if len(rankings_df) == 0:
        rankings_df = ranking_df.T
    else:
        rankings_df = pd.concat([rankings_df, ranking_df.T])
rankings_df = rankings_df.T


# convert into ranks
def _calculate_ranks(row: pd.Series) -> pd.Series:
    ranks = scipy.stats.rankdata(row, method="max", nan_policy="omit")
    result = pd.Series(ranks, index=row.index)
    return result


rankings_df = rankings_df.parallel_apply(_calculate_ranks, axis=0)

# heatmap
print(rankings_df)
destination_path = Path(
    config.OUTPUT_PATH
    / f"plots/leaderboard_single_hyperparameter_influence/all_together"
)

print(str(destination_path) + f".jpg")
set_seaborn_style(font_size=8)
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0
# plt.figure(figsize=set_matplotlib_size(fraction=10))

# calculate fraction based on length of keys
plt.figure(figsize=set_matplotlib_size(fraction=len(rankings_df.columns) / 6))

ax = sns.heatmap(rankings_df, annot=True, fmt="g")

ax.set_title(f"{hyperparameter_to_evaluate}")

# rankings_df.to_parquet(str(destination_path) + f".parquet")

plt.savefig(
    str(destination_path) + f".jpg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
exit(-1)
for hypothesis in [
    # "pearson",
    "kendall",
    # "spearman",
    # "kendall_unc_better_than_repr",
    # "same strategies - same rank"
    # "mm_better_lc_then_ent",
    # "random_similar",
    # "optimal best",
    # "quire similar"
    # "same strategy but in different frameworks behave similar"
]:
    # check how "well" the hypothesis can be found in the rankings!
    corr_data = rankings_df.corr(method=hypothesis)
    destination_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_{hypothesis}"
    )

    print(str(destination_path) + f".jpg")
    set_seaborn_style(font_size=8)
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = 1.0
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    plt.figure(figsize=set_matplotlib_size(fraction=len(corr_data.columns) / 6))

    ax = sns.heatmap(corr_data, annot=True, fmt=".2%", vmin=0, vmax=1)

    ax.set_title(f"{hyperparameter_to_evaluate}")

    corr_data.to_parquet(str(destination_path) + f".parquet")

    plt.savefig(
        str(destination_path) + f".jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
exit(-1)

for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
    ranking_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
    )
    ranking_df = pd.read_csv(ranking_path, index_col=0).T
    print(ranking_df)

    keys = {
        kkk: kkk.removeprefix(f"{hyperparameter_to_evaluate}: ")
        for kkk in ranking_df.columns
    }
    ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate == "EXP_LEARNER_MODEL":
        keys = {kkk: LEARNER_MODEL(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_BATCH_SIZE":
        keys = {kkk: int(kkk) for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_DATASET":
        keys = {kkk: DATASET(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate in ["random_seed_scenarios", "dataset_scenarios"]:
        custom_dict = {
            v: k
            for k, v in enumerate(
                sorted(
                    ranking_df.columns, key=lambda kkk: int(ast.literal_eval(kkk)[1])
                )
            )
        }
        ranking_df = ranking_df.sort_index(axis=0)
        ranking_df = ranking_df.sort_index(key=lambda x: x.map(custom_dict), axis=1)
    else:
        ranking_df = ranking_df.sort_index(axis=0)
        ranking_df = ranking_df.sort_index(axis=1)

    for hypothesis in [
        # "pearson",
        "kendall",
        "spearman",
        # "kendall_unc_better_than_repr",
        # "same strategies - same rank"
        # "mm_better_lc_then_ent",
        # "random_similar",
        # "optimal best",
        # "quire similar"
        # "same strategy but in different frameworks behave similar"
    ]:
        # check how "well" the hypothesis can be found in the rankings!
        corr_data = ranking_df.corr(method=hypothesis)
        destination_path = Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_{hypothesis}"
        )

        print(str(destination_path) + f".jpg")
        set_seaborn_style(font_size=8)
        mpl.rcParams["path.simplify"] = True
        mpl.rcParams["path.simplify_threshold"] = 1.0
        # plt.figure(figsize=set_matplotlib_size(fraction=10))

        # calculate fraction based on length of keys
        plt.figure(figsize=set_matplotlib_size(fraction=len(corr_data.columns) / 6))

        ax = sns.heatmap(corr_data, annot=True, fmt=".2%", vmin=0, vmax=1)

        ax.set_title(f"{hyperparameter_to_evaluate}")

        corr_data.to_parquet(str(destination_path) + f".parquet")

        plt.savefig(
            str(destination_path) + f".jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
