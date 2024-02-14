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

# (EXP_DATASET, EXP_BATCH_SIZE, EXP_LEARNER_MODEL, EXP_TRAIN_TEST_BUCKET_SIZE):17
cutoff_values: Dict[Tuple[str, int, int, int], float] = {}


def _do_stuff(file_name, config):

    font_size = 5.8
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": False,
        # "text.usetex": False,
        # "font.family": "times",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": font_size,
        "font.size": font_size,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "xtick.bottom": True,
        # "figure.autolayout": True,
    }

    plt.rcParams.update(tex_fonts)  # type: ignore

    sns.set_theme(rc={"figure.figsize": (12, 6)}, style="white", context="paper")
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_csv(metric_file)
    # print(df)

    small_done_df = done_workload_df.loc[
        done_workload_df["EXP_UNIQUE_ID"].isin(df.EXP_UNIQUE_ID)
    ]
    EXP_DATASET = DATASET(small_done_df["EXP_DATASET"].iloc[0])

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
    )["EXP_UNIQUE_ID"].apply(list)

    for k, v in grouped.items():
        interesting_rows = df.loc[df["EXP_UNIQUE_ID"].isin(v)]
        del interesting_rows["EXP_UNIQUE_ID"]

        grouped_cutoff_values_list = []

        """for ix, row in interesting_rows.iterrows():
            row = row.to_numpy()

            cutoff_values = defaultdict(lambda: 0)

            sliding_window_size = 20
            for sliding_window_indices in range(len(row), sliding_window_size - 1, -1):
                sliding_window_indices = range(
                    sliding_window_indices - sliding_window_size,
                    sliding_window_indices,
                )

                comparison_value = np.std(row[sliding_window_indices])

                for ttt in sorted([0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2]):
                    if comparison_value > ttt and cutoff_values[ttt] == 0:
                        cutoff_values[ttt] = sliding_window_indices[-1]
                # print(
                #    f"{sliding_window_indices[0]}:{sliding_window_indices[-1]}: {comparison_value:.2f} - {np.mean(row[sliding_window_indices]):.2f}"
                # )

            cutoff_values["111median"] = np.argmax(row > np.median(row))
            cutoff_values["111mean"] = np.argmax(row > np.mean(row))

            grouped_cutoff_values_list.append(cutoff_values)
            ax = sns.lineplot(row)

            for ci, cccc in enumerate(cutoff_values.items()):
                (ck, cv) = cccc
                ax.axvline(
                    x=cv,
                    lw=1,
                    alpha=0.5,
                    color=sns.color_palette(n_colors=len(cutoff_values.keys()))[ci],
                )
                text(cv, max(row), f"{ck}", rotation=90)
            ax.set_title(f"{EXP_DATASET.name} - {k}_{ix}")

            plt.savefig(
                f"plots_cutoffs/{EXP_DATASET.name}-{k}_{ix}.jpg",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.clf()

        avg_grouped_cutoff_values = defaultdict(list)

        for grouped_cutoff_values in grouped_cutoff_values_list:
            for kkkk, v in grouped_cutoff_values.items():
                avg_grouped_cutoff_values[kkkk].append(v)

        for kkkk, v in avg_grouped_cutoff_values.items():
            avg_grouped_cutoff_values[kkkk] = np.mean(v)
        """
        avg_grouped_cutoff_values = {}
        avg_grouped_cutoff_values["median"] = np.argmax(
            interesting_rows.mean().to_numpy() > np.median(interesting_rows)
        )
        avg_grouped_cutoff_values["mean"] = np.argmax(
            interesting_rows.mean().to_numpy() > np.mean(interesting_rows)
        )
        # avg_grouped_cutoff_values["median_of_mean"] = np.argmax(
        #    interesting_rows.mean().to_numpy() > np.median(interesting_rows.mean())
        # )
        melted_data = interesting_rows.melt(var_name="x", value_name="val")
        ax = sns.lineplot(data=melted_data, x="x", y="val")

        for ci, cccc in enumerate(avg_grouped_cutoff_values.items()):
            (ck, cv) = cccc
            ax.axvline(
                x=cv,
                lw=1,
                alpha=0.5,
                color=sns.color_palette(n_colors=len(avg_grouped_cutoff_values.keys()))[
                    ci
                ],
            )
            text(cv, interesting_rows.mean().max(), f"{ck}", rotation=90)

        ax.set_title(f"{EXP_DATASET.name} - {k}")
        plt.savefig(
            f"plots_single_cutoffs/{EXP_DATASET.name}-{k}_grouped.jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()


# for EXP_DATASET in config.EXP_GRID_DATASET:
# if EXP_DATASET.name in ["Iris", "wine_origin", "glass", "parkinsons"]:
#    continue
glob_list = [
    f
    for f in glob.glob(
        str(config.OUTPUT_PATH) + f"/ALIPY_RANDOM/*/accuracy.csv.xz",
        recursive=True,
    )
]

# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name, config) for file_name in glob_list
)
