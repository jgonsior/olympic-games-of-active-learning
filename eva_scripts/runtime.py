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


runtime_metric = "query_selection_time"


if not Path(config.CORRELATION_TS_PATH / f"{runtime_metric}.parquet").exists():
    unsorted_f = config.CORRELATION_TS_PATH / f"{runtime_metric}.unsorted.csv"
    unparqueted_f = config.CORRELATION_TS_PATH / f"{runtime_metric}.to_parquet.csv"

    if not unsorted_f.exists() and not unparqueted_f.exists():
        log_and_time("Create selected indices ts")
        create_fingerprint_joined_timeseries_csv_files(
            metric_names=[runtime_metric], config=config
        )

    if not unparqueted_f.exists():
        log_and_time("Created, now sorting")
        command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {unsorted_f} -o {config.CORRELATION_TS_PATH}/{runtime_metric}.to_parquet.csv"
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

    f = Path(config.CORRELATION_TS_PATH / f"{runtime_metric}.parquet")
    ts.to_parquet(f)
    unparqueted_f.unlink()

ts = pd.read_parquet(
    config.CORRELATION_TS_PATH / f"{runtime_metric}.parquet",
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
print(f"{runtime_metric}.parquet")
print(ts)

fingerprint_cols = list(ts.columns)
fingerprint_cols.remove("metric_value")
fingerprint_cols.remove("EXP_STRATEGY")

ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
    lambda row: "_".join([str(rrr) for rrr in row]), axis=1
)


for fg_col in fingerprint_cols:
    del ts[fg_col]

log_and_time("Done fingerprinting")
print(ts)

shared_fingerprints = None
for target_value in ts["EXP_STRATEGY"].unique():
    tmp_fingerprints = set(
        ts.loc[ts["EXP_STRATEGY"] == target_value]["fingerprint"].to_list()
    )

    if shared_fingerprints is None:
        print(target_value)
        shared_fingerprints = tmp_fingerprints
    else:
        print(f"{target_value}: {len(shared_fingerprints)}")
        shared_fingerprints = shared_fingerprints.intersection(tmp_fingerprints)

log_and_time(f"Done calculating shared fingerprints - {len(shared_fingerprints)}")

ts = ts.loc[(ts["fingerprint"].isin(shared_fingerprints))]

del ts["fingerprint"]


destination_path = Path(config.OUTPUT_PATH / f"plots/runtime")
destination_path.mkdir(exist_ok=True, parents=True)

print(ts)

print(destination_path / f"{runtime_metric}.jpg")

set_seaborn_style(font_size=8)
# plt.figure(figsize=set_matplotlib_size(fraction=10))

# calculate fraction based on length of keys
# plt.figure(figsize=set_matplotlib_size(fraction=len(ts.columns) / 6))
ax = sns.barplot(data=ts, y="EXP_STRATEGY", x="metric_value")
ax.set(ylabel=None)
ax.set_xscale("log")
for container in ax.containers:
    ax.bar_label(container, padding=10, fmt="%.2g")


ax.set_title(f"Runtimes: {runtime_metric}")

ts.to_parquet(destination_path / f"{runtime_metric}.parquet")

plt.savefig(
    destination_path / f"{runtime_metric}.jpg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
# goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
