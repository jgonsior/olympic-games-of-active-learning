from ast import mod
import itertools
from pathlib import Path
import subprocess
import sys
import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pandas as pd
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    get_done_workload_joined_with_metric,
    get_done_workload_joined_with_multiple_metrics,
    get_glob_list,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_seaborn_style, set_matplotlib_size
import multiprocessing

sys.dont_write_bytecode = True

from misc.config import Config
import numpy as np
import seaborn as sns
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=multiprocessing.cpu_count(), progress_bar=True)
config = Config()


for modus in ["standard", "extended", "auc"]:
    standard_metrics = [
        "accuracy",
        "weighted_recall",
        "macro_f1-score",
        "macro_precision",
        "macro_recall",
        "weighted_f1-score",
        "weighted_precision",
    ]

    if modus == "extended" or modus == "auc":
        standard_metrics = ["macro_f1-score"]
        variant_prefixe = [
            "biggest_drop_per_",
            "nr_decreasing_al_cycles_per_",
        ]

        original_standard_metrics = standard_metrics.copy()
        for vp in variant_prefixe:
            standard_metrics = [
                *standard_metrics,
                *[vp + sss for sss in original_standard_metrics],
            ]

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
        command = f"sort --parallel {multiprocessing.cpu_count()} {f} -o {f.split('.')[0]}.csv"
        # print(command)
        subprocess.run(command, shell=True, text=True)
        Path(f).unlink()

    log_and_time("computing intersection")
    shared_unique_ids = None

    for sm in standard_metrics:
        ts = pd.read_csv(
            config.CORRELATION_TS_PATH / f"{sm}.csv",
            header=None,
            index_col=False,
            delimiter=",",
            usecols=[0],
        )
        if shared_unique_ids is None:
            shared_unique_ids = set(ts.iloc[:, 0].to_list())
        else:
            shared_unique_ids = shared_unique_ids.intersection(
                set(ts.iloc[:, 0].to_list())
            )

    log_and_time("Reading in ts csv files")
    timeseriesses = []
    for sm in standard_metrics:
        ts = pd.read_csv(
            config.CORRELATION_TS_PATH / f"{sm}.csv",
            header=None,
            index_col=False,
            delimiter=",",
            dtype={0: str, 1: np.float32},
            usecols=[0, 8],
        )
        ts = ts.loc[ts[0].isin(shared_unique_ids)]
        timeseriesses.append(ts.iloc[:, 1].values)
    timeseriesses = np.array(timeseriesses)

    log_and_time("numpied")
    corrmat = np.corrcoef(timeseriesses)

    log_and_time("corrmatted")
    save_correlation_plot(
        data=corrmat, title=modus, keys=standard_metrics, config=config, total=True
    )
