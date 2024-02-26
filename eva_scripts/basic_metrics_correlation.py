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
                "plateu_auc_",
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

    print(modus)
    create_fingerprint_joined_timeseries_csv_files(standard_metrics, config)

    print("Sorting files")
    command = (
        "find "
        + str(config.CORRELATION_TS_PATH)
        + "/ -type f -exec sort --parallel "
        + str(multiprocessing.cpu_count())
        + " {} -o {} \;"
    )
    subprocess.run(command, shell=True, text=True)

    print("Reading in csv files")

    timeseriesses = []
    for sm in standard_metrics:
        ts = np.loadtxt(
            config.CORRELATION_TS_PATH / f"{sm}.csv",
            delimiter=",",
            dtype="float32",
            usecols=1,
        )
        timeseriesses.append(ts)

    timeseriesses = np.array(timeseriesses)
    print("numpied")
    corrmat = np.corrcoef(timeseriesses)

    print("corrmatted")
    save_correlation_plot(
        data=corrmat, title=modus, keys=standard_metrics, config=config, total=True
    )
