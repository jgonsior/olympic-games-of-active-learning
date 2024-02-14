import csv
from math import e
from pathlib import Path
from re import I
import sys
import glob
from typing import Dict, Tuple

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import DATASET


sys.dont_write_bytecode = True

from misc.config import Config

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
        by=["EXP_BATCH_SIZE", "EXP_LEARNER_MODEL", "EXP_TRAIN_TEST_BUCKET_SIZE"]
    )["EXP_UNIQUE_ID"].apply(list)
    # print(grouped)

    for k, v in grouped.items():
        # print(k)

        interesting_rows = df.loc[df["EXP_UNIQUE_ID"].isin(v)]
        del interesting_rows["EXP_UNIQUE_ID"]
        # print(interesting_rows)
        melted_data = interesting_rows.melt(var_name="x", value_name="val")
        ax = sns.lineplot(data=melted_data, x="x", y="val")

        ax.set_title(f"{EXP_DATASET.name} - {k}")
        plt.savefig(
            f"plots_cutoffs/{EXP_DATASET.name}-{k}.jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()


for EXP_DATASET in config.EXP_GRID_DATASET:
    # if EXP_DATASET.name in ["Iris", "wine_origin"]:
    #    continue
    glob_list = [
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH)
            + f"/ALIPY_RANDOM/{EXP_DATASET.name}/accuracy.csv.xz",
            recursive=True,
        )
    ]

    Parallel(n_jobs=1, verbose=10)(
        # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config)
        for file_name in glob_list
    )
