from collections import defaultdict
import csv
from itertools import count
from math import e
import multiprocessing
from pathlib import Path
from re import I, S
import re
import sys
import glob
from typing import Dict, Tuple

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib.hatch import VerticalHatch
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DATASET
from matplotlib.pyplot import text
from statsmodels.tsa.stattools import adfuller, kpss
from misc.plotting import set_seaborn_style

sys.dont_write_bytecode = True

from misc.config import Config


config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
print(done_workload_df.keys())

ramp_plateau_results_file = config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH


if not ramp_plateau_results_file.exists():
    with open(ramp_plateau_results_file, "a") as f:
        w = csv.DictWriter(f, fieldnames=["metric_file"])
        w.writeheader()


# read in accuracy files from random strategy, average them/smooth them
# check all buckets of 5 AL cycles for stationary
# when change from "stationary" to "non-stationary" --> we have a slope!

# (EXP_DATASET, EXP_BATCH_SIZE, EXP_LEARNER_MODEL, EXP_TRAIN_TEST_BUCKET_SIZE, EXP_START_POINT):17
cutoff_values: Dict[Tuple[str, int, int, int, int], float] = {}


def _do_stuff(file_name, config):
    set_seaborn_style()
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_csv(metric_file)
    # print(df)

    small_done_df = done_workload_df.loc[
        done_workload_df["EXP_UNIQUE_ID"].isin(df.EXP_UNIQUE_ID)
    ]
    EXP_DATASET = DATASET(small_done_df["EXP_DATASET"].iloc[0])

    if EXP_DATASET in [
        DATASET["vowel"],
        DATASET["soybean"],
        DATASET["statlog_vehicle"],
    ]:
        return

    del small_done_df["EXP_STRATEGY"]
    del small_done_df["EXP_DATASET"]
    del small_done_df["EXP_RANDOM_SEED"]
    del small_done_df["EXP_NUM_QUERIES"]

    grouped = small_done_df.groupby(
        by=[
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
            "EXP_START_POINT",
        ]
    )["EXP_UNIQUE_ID"].apply(lambda rrr: rrr)

    for k, EXP_UNIQUE_ID in grouped.items():
        current_row = df.loc[df["EXP_UNIQUE_ID"] == EXP_UNIQUE_ID]
        del current_row["EXP_UNIQUE_ID"]
        current_row_np = current_row.to_numpy()[0]
        print(current_row)

        current_cutoff_values = {}
        current_cutoff_values["median"] = np.argmax(
            current_row_np > np.median(current_row)
        )
        current_cutoff_values["mean"] = np.argmax(
            current_row_np > np.mean(current_row_np)
        )

        for window_size in range(10, len(current_row_np)):
            window = current_row_np[-window_size:]

            # print(window)

            dftest = adfuller(window, autolag="t-stat", regression="ctt")

            if dftest[1] > 0.05:
                res = "non stat"
            else:
                res = "    stat"
            # print(window.tolist())
            print(
                f"{len(window)} : {res} {dftest[0]:.2f} {dftest[1]:.2f} {dftest[4]['5%']:.2f} "
            )

        melted_data = current_row.melt(var_name="x", value_name="val")
        ax = sns.lineplot(data=melted_data, x="x", y="val")

        for ci, cccc in enumerate(current_cutoff_values.items()):
            (ck, cv) = cccc
            ax.axvline(
                x=cv,
                lw=1,
                alpha=0.5,
                color=sns.color_palette(n_colors=len(current_cutoff_values.keys()))[ci],
            )
            text(cv, current_row.mean().max(), f"{ck}", rotation=90)

        ax.set_title(f"{EXP_DATASET.name} - {k}")
        # plt.show()
        exit(-1)

        plt.savefig(
            f"plots_single_cutoffs/{EXP_DATASET.name}-{k}_grouped.jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()


glob_list = [
    f
    for f in glob.glob(
        str(config.OUTPUT_PATH) + f"/ALIPY_RANDOM/*/accuracy.csv.xz",
        recursive=True,
    )
]

Parallel(n_jobs=1, verbose=10)(
    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name, config)
    for file_name in glob_list
)
