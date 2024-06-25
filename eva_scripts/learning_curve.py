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

standard_metric = "weighted_f1-score"


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
        "EXP_BATCH_SIZE",
        "EXP_LEARNER_MODEL",
        "EXP_TRAIN_TEST_BUCKET_SIZE",
        "ix",
        # "EXP_UNIQUE_ID_ix",
        "metric_value",
    ],
)
print(ts)
ts = ts.loc[
    (ts["EXP_DATASET"] == 2)
    & (ts["EXP_BATCH_SIZE"] == 1)
    & (ts["EXP_LEARNER_MODEL"] == 1)
    & (ts["EXP_TRAIN_TEST_BUCKET_SIZE"] == 1)
]
print(f"{standard_metric}.parquet")
print(ts)


destination_path = Path(config.OUTPUT_PATH / f"plots/single_learning_curve")
destination_path.mkdir(exist_ok=True, parents=True)

ts["EXP_STRATEGY"] = ts["EXP_STRATEGY"].parallel_apply(
    lambda xxx: AL_STRATEGY(int(xxx)).name
)

set_seaborn_style(font_size=8)
# plt.figure(figsize=set_matplotlib_size(fraction=10))

# calculate fraction based on length of keys
plt.figure(figsize=set_matplotlib_size(fraction=len(ts.columns) / 6))
ax = sns.lineplot(ts, x="ix", y="metric_value", hue="EXP_STRATEGY")

ax.set_title(f"Learning Curve: {standard_metric}")

ts.to_parquet(destination_path / f"{standard_metric}.parquet")

plt.savefig(
    destination_path / f"{standard_metric}.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
# goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
