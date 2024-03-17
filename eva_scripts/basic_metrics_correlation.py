from pathlib import Path
import subprocess
import sys
import glob
import pandas as pd
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
import multiprocessing

sys.dont_write_bytecode = True

from misc.config import Config
import numpy as np
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=multiprocessing.cpu_count(), progress_bar=True)
config = Config()


#  for modus in ["auc", "extended", "standard"]:
for modus in ["standard", "extended", "auc", "auc2"]:
    standard_metrics = [
        "accuracy",
        "weighted_recall",
        "macro_f1-score",
        "macro_precision",
        "macro_recall",
        "weighted_f1-score",
        "weighted_precision",
    ]

    if modus == "extended" or modus == "auc" or modus == "auc2":
        #  standard_metrics = ["macro_f1-score"]

        if modus == "auc":
            standard_metrics = ["weighted_f1-score", "macro_f1-score", "accuracy"]
        elif modus == "auc2":
            standard_metrics = ["weighted_f1-score"]
        #  standard_metrics = ["accuracy"]
        variant_prefixe = [
            "biggest_drop_per_",
            "nr_decreasing_al_cycles_per_",
        ]
        #  TODO variant_prefixe klappt ni

        original_standard_metrics = standard_metrics.copy()
        for vp in variant_prefixe:
            standard_metrics = [
                *standard_metrics,
                *[vp + sss for sss in original_standard_metrics],
            ]
        print(standard_metrics)
        if modus == "auc":
            auc_prefixe = [
                "final_value_",
                "first_5_",
                "full_auc_",
                "last_5_",
                "learning_stability_5_",
                "learning_stability_10_",
                "ramp_up_auc_",
                "plateau_auc_",
            ]

            original_standard_metrics = standard_metrics.copy()
            standard_metrics = []
            for vp in auc_prefixe:
                standard_metrics = [
                    *standard_metrics,
                    *[vp + sss for sss in original_standard_metrics],
                ]

        standard_metrics = [
            *standard_metrics,
            *[sss + "_time_lag" for sss in standard_metrics],
        ]

    log_and_time(modus)
    create_fingerprint_joined_timeseries_csv_files(standard_metrics, config)
    log_and_time("Sorting files")

    for f in glob.glob(
        str(config.CORRELATION_TS_PATH) + f"/*.unsorted.csv", recursive=True
    ):
        command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {f} -o {f.split('.')[0]}.to_parquet.csv"
        # print(command)
        subprocess.run(command, shell=True, text=True)
        Path(f).unlink()

    log_and_time("Parquetting files")

    for f in glob.glob(
        str(config.CORRELATION_TS_PATH) + f"/*.to_parquet.csv", recursive=True
    ):
        ts = pd.read_csv(
            f,
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
        ts.to_parquet(f"{f.split('.')[0]}.parquet")
        Path(f).unlink()

    log_and_time("computing intersection")
    shared_unique_ids = None

    for sm in standard_metrics:
        ts = pd.read_parquet(
            config.CORRELATION_TS_PATH / f"{sm}.parquet",
            columns=["EXP_UNIQUE_ID_ix"],
        )
        if shared_unique_ids is None:
            shared_unique_ids = set(ts["EXP_UNIQUE_ID_ix"].to_list())
        else:
            shared_unique_ids = shared_unique_ids.intersection(
                set(ts["EXP_UNIQUE_ID_ix"].to_list())
            )

    log_and_time("Reading in ts csv files")
    timeseriesses = []
    for sm in standard_metrics:
        log_and_time(f"Reading in {sm}")
        ts = pd.read_parquet(
            config.CORRELATION_TS_PATH / f"{sm}.parquet",
            columns=["EXP_UNIQUE_ID_ix", "metric_value"],
        )
        print(ts)
        ts = ts.loc[ts["EXP_UNIQUE_ID_ix"].isin(shared_unique_ids)]
        timeseriesses.append(ts["metric_value"].values)
    timeseriesses = np.array(timeseriesses)

    log_and_time("numpied")
    corrmat = np.corrcoef(timeseriesses)

    log_and_time("corrmatted")
    save_correlation_plot(
        data=corrmat, title=modus, keys=standard_metrics, config=config, total=True
    )
