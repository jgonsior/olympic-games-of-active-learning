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


for modus in ["standard"]:  # ["extended", "standard", "auc"]:
    standard_metrics = [
        "accuracy",
        "weighted_recall",
        "macro_f1-score",
        "macro_precision",
        "macro_recall",
        "weighted_f1-score",
        "weighted_precision",
        "weighted_recall",
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
        "find " + str(config.CORRELATION_TS_PATH) + "/ -type f -exec sort {} -o {} \;"
    )
    subprocess.run(command, shell=True, text=True)

    exit(-1)
    del df["EXP_UNIQUE_ID"]

    print(df)

    non_al_cycle_keys = [
        "EXP_DATASET",
        "EXP_STRATEGY",
        "EXP_BATCH_SIZE",
        "EXP_LEARNER_MODEL",
        "EXP_TRAIN_TEST_BUCKET_SIZE",
        "EXP_START_POINT",
    ]

    metric_keys = [kkk for kkk in df.columns if kkk not in non_al_cycle_keys]

    # replace non_al_cycle_keys by single string fingerprint as key
    df["fingerprint"] = df[non_al_cycle_keys].parallel_apply(
        lambda row: "_".join(row.values.astype(str)),
        axis=1,
    )
    print("fingerprints")

    for non_al_cycle_key in non_al_cycle_keys:
        del df[non_al_cycle_key]

    print(df)
    df = df.melt(id_vars=["metric_name", "fingerprint"], value_vars=metric_keys)
    print(df)

    print("melted")
    df["fingerprint"] = df[["fingerprint", "variable"]].parallel_apply(
        lambda row: "_".join(row.values), axis=1
    )
    print(df)
    exit(-1)

    del df["variable"]

    df.dropna(inplace=True)

    df = df.pivot(
        index="fingerprint", columns="metric_name", values="value"
    ).reset_index()
    print("pivoted")
    df.columns.name = None
    df.index = df["fingerprint"]
    del df["fingerprint"]
    df.dropna(inplace=True)

    # exit(-1)
    # print(df.corr(method="spearman"))
    # print(df.corr())

    data = df.to_numpy()
    print("numpied")
    corrmat = np.corrcoef(data.T)
    print("corrmatted")
    save_correlation_plot(
        data=corrmat, title=modus, keys=df.columns.to_list(), config=config, total=True
    )
    exit(-1)
