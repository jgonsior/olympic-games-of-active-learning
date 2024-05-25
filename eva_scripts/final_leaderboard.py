import multiprocessing
import subprocess
import sys
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

orig_standard_metric = "weighted_f1-score"


# what are the different ways for leaderboard?
# aggregation over aggregation
# ranking only (pearson vs spearman)

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
            # "EXP_START_POINT",
            # "EXP_BATCH_SIZE",
            # "EXP_LEARNER_MODEL",
            # "EXP_TRAIN_TEST_BUCKET_SIZE",
            # "ix",
            # "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )
    ts = (
        ts.groupby(by=["EXP_DATASET", "EXP_STRATEGY"])["metric_value"]
        .apply(lambda lll: np.array([llllll for llllll in lll]).flatten())
        .reset_index()
    )
    ts = ts.pivot(index="EXP_DATASET", columns="EXP_STRATEGY", values="metric_value")
    ts = ts.parallel_apply(np.mean)
    print(ts)

    # goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
