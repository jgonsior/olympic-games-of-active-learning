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

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

hyperparameters_to_evaluate = [
    # "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    # "EXP_DATASET",
    # "EXP_TRAIN_TEST_BUCKET_SIZE",
    # "EXP_START_POINT",
]

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

    ranking_df = ranking_df.sort_index(axis=0)
    ranking_df = ranking_df.sort_index(axis=1)
    print(ranking_df)

    for corr_method in ["spearman", "kendall"]:
        corr_data = ranking_df.corr(method=corr_method)

        destination_path = Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_{corr_method}"
        )

        print(str(destination_path) + f".jpg")
        set_seaborn_style(font_size=8)
        mpl.rcParams["path.simplify"] = True
        mpl.rcParams["path.simplify_threshold"] = 1.0
        # plt.figure(figsize=set_matplotlib_size(fraction=10))

        # calculate fraction based on length of keys
        plt.figure(figsize=set_matplotlib_size(fraction=len(corr_data.columns) / 6))

        ax = sns.heatmap(corr_data, annot=True, fmt=".2%")

        ax.set_title(f": {hyperparameter_to_evaluate}")

        corr_data.to_parquet(str(destination_path) + f".parquet")

        plt.savefig(
            str(destination_path) + f".jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
