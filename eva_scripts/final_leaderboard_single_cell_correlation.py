import multiprocessing
import subprocess
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
)
from misc.plotting import _rename_strategy, set_seaborn_style

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=False, use_memory_fs=False
)

default_standard_metric = "full_auc_weighted_f1-score"
grid_type = "sparse"  # dense is not supported by this script!
rank_or_percentage = "dataset_normalized_percentages"
interpolation = "average_of_same_strategy"

hyperparameters_to_evaluate = [
    # "standard_metric",
    # "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    # "EXP_DATASET",
    # "EXP_TRAIN_TEST_BUCKET_SIZE",
    # "EXP_START_POINT",
    # "auc_metric",
]


def read_or_create_ts(metric_name) -> pd.DataFrame:
    if not Path(config.CORRELATION_TS_PATH / f"{metric_name}.parquet").exists():
        unsorted_f = config.CORRELATION_TS_PATH / f"{metric_name}.unsorted.csv"
        unparqueted_f = config.CORRELATION_TS_PATH / f"{metric_name}.to_parquet.csv"

        if not unsorted_f.exists() and not unparqueted_f.exists():
            log_and_time("Create selected indices ts")
            create_fingerprint_joined_timeseries_csv_files(
                metric_names=[metric_name], config=config
            )

        if not unparqueted_f.exists():
            log_and_time("Created, now sorting")
            command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {unsorted_f} -o {config.CORRELATION_TS_PATH}/{metric_name}.to_parquet.csv"
            print(command)
            subprocess.run(command, shell=True, text=True)
            unsorted_f.unlink()

        log_and_time("sorted, now parqueting")
        ts = pd.read_csv(
            unparqueted_f,
            header=None,
            index_col=False,
            delimiter=",",
            names=[
                "EXP_DATASET",
                "EXP_STRATEGY",
                "EXP_START_POINT",
                "EXP_BATCH_SIZE",
                "EXP_LEARNER_MODEL",
                "EXP_TRAIN_TEST_BUCKET_SIZE",
                "ix",
                "EXP_UNIQUE_ID_ix",
                "metric_value",
            ],
        )
        f = Path(config.CORRELATION_TS_PATH / f"{metric_name}.parquet")
        ts.to_parquet(f)
        unparqueted_f.unlink()

    ts = pd.read_parquet(
        config.CORRELATION_TS_PATH / f"{metric_name}.parquet",
        columns=[
            "EXP_DATASET",
            "EXP_STRATEGY",
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
            # "ix",
            # "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )
    return ts


for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
    ranking_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
    )
    print(ranking_path)
    ranking_df = pd.read_csv(ranking_path, index_col=0)
    ranking_df.rename(columns=_rename_strategy, inplace=True)

    ranking_df = ranking_df.T

    keys = {
        kkk: kkk.removeprefix(f"{hyperparameter_to_evaluate}: ")
        for kkk in ranking_df.columns
    }
    ranking_df.rename(columns=keys, inplace=True)

    keys = {kkk: int(kkk) for kkk in ranking_df.columns}
    ranking_df.rename(columns=keys, inplace=True)
    ranking_df = ranking_df.sort_index(axis=0)
    ranking_df = ranking_df.sort_index(axis=1)
    print(ranking_df)

    destination_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_single_correlation"
    )

    print(str(destination_path) + f".jpg")
    set_seaborn_style(font_size=8)
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = 1.0
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    # plt.figure(figsize=set_matplotlib_size())  # fraction=len(corr_data.columns) / 6))

    ax = sns.scatterplot(data=ranking_df, x=1, y=100)
    ax = sns.scatterplot(data=ranking_df, x=20, y=10)
    ax = sns.scatterplot(data=ranking_df, x=20, y=50)
    ax = sns.scatterplot(data=ranking_df, x=50, y=100)
    ax = sns.scatterplot(data=ranking_df, x=5, y=1)

    ax.set_title(f"{hyperparameter_to_evaluate}")
    plt.legend([], [], frameon=False)
    plt.savefig(
        str(destination_path) + f".jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.clf()
    exit(-1)
