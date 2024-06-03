import multiprocessing
import subprocess
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size, set_seaborn_style
from resources.data_types import AL_STRATEGY
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

orig_standard_metric = "weighted_f1-score"


# what are the different ways for leaderboard?
# DONE aggregation over aggregation
# ranking only (spearman instead of pearson)
# normalize datasets first

for auc_prefix in [
    "full_auc_",
    "first_5_",
    "last_5_",
    "ramp_up_auc_",
    "plateau_auc_",
    "final_value_",
]:
    standard_metric = auc_prefix + orig_standard_metric

    log_and_time(f"Calculating for {standard_metric}")

    if not Path(config.CORRELATION_TS_PATH / f"{standard_metric}.parquet").exists():
        unsorted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.unsorted.csv"
        unparqueted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.to_parquet.csv"

        if not unsorted_f.exists() and not unparqueted_f.exists():
            log_and_time("Create selected indices ts")
            create_fingerprint_joined_timeseries_csv_files(
                metric_names=[standard_metric], config=config
            )

        if not unparqueted_f.exists():
            log_and_time("Created, now sorting")
            command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {unsorted_f} -o {config.CORRELATION_TS_PATH}/{standard_metric}.to_parquet.csv"
            print(command)
            subprocess.run(command, shell=True, text=True)
            unsorted_f.unlink()
        print(unparqueted_f)
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
        print(ts["metric_value"])
        ts["metric_value"] = ts["metric_value"].apply(
            lambda xxx: (
                np.fromstring(
                    xxx.removeprefix("[").removesuffix("]"),
                    dtype=np.int32,
                    sep=" ",
                )
            )
        )

        f = Path(config.CORRELATION_TS_PATH / f"{standard_metric}.parquet")
        ts.to_parquet(f)
        unparqueted_f.unlink()

    ts = pd.read_parquet(
        config.CORRELATION_TS_PATH / f"{standard_metric}.parquet",
        columns=[
            "EXP_DATASET",
            "EXP_STRATEGY",
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
            "ix",
            # "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )

    print(ts)
    fingerprint_cols = list(ts.columns)
    fingerprint_cols.remove("metric_value")
    fingerprint_cols.remove("EXP_DATASET")
    fingerprint_cols.remove("EXP_STRATEGY")

    ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
        lambda row: "_".join([str(rrr) for rrr in row]), axis=1
    )

    ts["dataset_strategy"] = ts[["EXP_DATASET", "EXP_STRATEGY"]].parallel_apply(
        lambda row: "_".join([str(rrr) for rrr in row]), axis=1
    )

    for fg_col in fingerprint_cols:
        del ts[fg_col]

    log_and_time("Done fingerprinting")
    print(ts)

    shared_fingerprints = None
    for target_value in ts["dataset_strategy"].unique():
        tmp_fingerprints = set(
            ts.loc[ts["dataset_strategy"] == target_value]["fingerprint"].to_list()
        )

        if shared_fingerprints is None:
            shared_fingerprints = tmp_fingerprints
        else:
            shared_fingerprints = shared_fingerprints.intersection(tmp_fingerprints)

    log_and_time(f"Done calculating shared fingerprints - {len(shared_fingerprints)}")

    ts = ts.loc[(ts["fingerprint"].isin(shared_fingerprints))]

    print(ts)
    del ts["dataset_strategy"]
    del ts["fingerprint"]

    # @todo shared fingerprints hier betrachten!
    # was mache ich mit lücken? z. B. quire :/
    # lücken wegen error -> alles weg?
    # lücken wegen timeout -> 0%? oder so viel wie random bei iteration 0 hat?

    ts = (
        ts.groupby(by=["EXP_DATASET", "EXP_STRATEGY"])["metric_value"]
        .apply(lambda lll: np.array([llllll for llllll in lll]).flatten())
        .reset_index()
    )
    ts = ts.pivot(index="EXP_DATASET", columns="EXP_STRATEGY", values="metric_value")
    ts = ts.parallel_applymap(np.mean)
    # ts = ts.parallel_applymap(np.median)

    ts.columns = [AL_STRATEGY(int(kkk)).name for kkk in ts.columns]

    destination_path = Path(config.OUTPUT_PATH / f"plots/final_leaderboard")
    destination_path.mkdir(exist_ok=True, parents=True)

    ts = ts.set_index([[DATASET(int(kkk)).name for kkk in ts.index]])

    ts = ts.T
    ts.loc[:, "Total"] = ts.mean(axis=1)
    ts.sort_values(by=["Total"], inplace=True)
    ts = ts.T
    print(ts)

    print(destination_path / f"{standard_metric}.jpg")
    set_seaborn_style(font_size=8)
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    plt.figure(figsize=set_matplotlib_size(fraction=len(ts.columns) / 6))
    ax = sns.heatmap(ts, annot=True, fmt=".2%")

    ax.set_title(f"Final leaderboard: {standard_metric}")

    ts.to_parquet(destination_path / f"{standard_metric}.parquet")

    plt.savefig(
        destination_path / f"{standard_metric}.jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    # goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
