from collections import OrderedDict, defaultdict
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
from matplotlib import lines, pyplot as plt
from matplotlib.hatch import VerticalHatch
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DATASET
from matplotlib.pyplot import text
from statsmodels.tsa.stattools import adfuller, kpss
from misc.plotting import set_seaborn_style
import Rbeast as rb

sys.dont_write_bytecode = True

from misc.config import Config
import ruptures as rpt

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
    set_seaborn_style(font_size=10)
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_csv(metric_file)
    # print(df)

    small_done_df = done_workload_df.loc[
        done_workload_df["EXP_UNIQUE_ID"].isin(df.EXP_UNIQUE_ID)
    ]
    EXP_DATASET = DATASET(small_done_df["EXP_DATASET"].iloc[0])

    """if EXP_DATASET in [
        DATASET["vowel"],
        DATASET["soybean"],
        DATASET["statlog_vehicle"],
    ]:
        return
    
    if EXP_DATASET not in [DATASET["letter"], DATASET["MiceProtein"]]:
        return
    """
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

    for row_count, (k, EXP_UNIQUE_ID) in enumerate(grouped.sample(frac=1).items()):
        current_row = df.loc[df["EXP_UNIQUE_ID"] == EXP_UNIQUE_ID]
        del current_row["EXP_UNIQUE_ID"]
        current_row_np = current_row.to_numpy()[0]
        current_row_np = current_row_np[~np.isnan(current_row_np)]
        # print(current_row_np)

        current_cutoff_values = {}
        current_cutoff_values["dyn_l2"] = (
            rpt.Dynp(model="l2").fit(current_row_np).predict(n_bkps=1)[0]
        )

        beast_res = rb.beast(current_row_np, season="none", print_options=0)

        for cp_i, cp in enumerate(beast_res.trend.cp[:1]):
            if np.isnan(cp):
                continue
            current_cutoff_values[f"cp_{cp_i}"] = cp

        # current_cutoff_values["mean"] = np.argmax(
        #    current_row_np > np.mean(current_row_np)
        # )

        """for window_size in range(5, len(current_row_np)):
            window = current_row_np[-window_size:]
            mk_result = mk.hamed_rao_modification_test(window, alpha=0.1)[:2]
            mk_result1 = mk.original_test(window, alpha=0.1)[:2]
            mk_result2 = mk.yue_wang_modification_test(window, alpha=0.1)[:2]
            mk_result3 = mk.pre_whitening_modification_test(window, alpha=0.1)[:2]
            mk_result4 = mk.trend_free_pre_whitening_modification_test(
                window, alpha=0.1
            )[:2]
            mk_result5 = mk.regional_test(window, alpha=0.1)[:2]
            # print(f"{len(window)}: {mk_result}")
            if mk_result[0] == "increasing" and mk_result[1] == True:
                current_cutoff_values["t1"] = len(current_row_np) - window_size
            if mk_result1[0] == "increasing" and mk_result1[1] == True:
                current_cutoff_values["t2"] = len(current_row_np) - window_size
            if mk_result2[0] == "increasing" and mk_result2[1] == True:
                current_cutoff_values["t3"] = len(current_row_np) - window_size
            if mk_result3[0] == "increasing" and mk_result3[1] == True:
                current_cutoff_values["t4"] = len(current_row_np) - window_size
            if mk_result4[0] == "increasing" and mk_result4[1] == True:
                current_cutoff_values["t5"] = len(current_row_np) - window_size

            if mk_result5[0] == "increasing" and mk_result5[1] == True:
                current_cutoff_values["regional_test"] = (
                    len(current_row_np) - window_size
                )
                # print(current_cutoff_values["trend"])
                # break

            if len(np.unique(window)) == 1:
                continue

            dftest = adfuller(window, autolag="t-stat", regression="ctt")

            if dftest[1] > 0.05:
                res = "non stat"
            else:
                res = "    stat"
            # print(window.tolist())
            print(
                f"{len(window)} : {res} {dftest[0]:.2f} {dftest[1]:.2f} {dftest[4]['5%']:.2f} "
            )"""
        ax = sns.lineplot(current_row_np)

        linestyles = [
            lll
            for lll in OrderedDict(
                [
                    ("solid", (0, ())),
                    ("loosely dotted", (0, (1, 10))),
                    ("dotted", (0, (1, 5))),
                    ("densely dotted", (0, (1, 1))),
                    ("loosely dashed", (0, (5, 10))),
                    ("dashed", (0, (5, 5))),
                    ("densely dashed", (0, (5, 1))),
                    ("loosely dashdotted", (0, (3, 10, 1, 10))),
                    ("dashdotted", (0, (3, 5, 1, 5))),
                    ("densely dashdotted", (0, (3, 1, 1, 1))),
                    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
                    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
                    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
                ]
            ).values()
        ]
        linestyles = ["-", "--", "-.", ":"]

        for ci, cccc in enumerate(current_cutoff_values.items()):
            (ck, cv) = cccc
            ax.axvline(
                x=cv,
                lw=3,
                alpha=0.5,
                color=sns.color_palette(n_colors=len(current_cutoff_values.keys()))[ci],
                linestyle=linestyles[ci],
            )
            text(cv, current_row.mean().max(), f"{ck}: {cv}", rotation=90, fontsize=12)

        ax.set_title(f"{EXP_DATASET.name} - {k}")
        # plt.show()
        # exit(-1)
        plt.savefig(
            f"plots_single/{EXP_DATASET.name}-{k}_grouped.jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.clf()
        if row_count >= 3:
            return
        continue
        exit(-1)


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
