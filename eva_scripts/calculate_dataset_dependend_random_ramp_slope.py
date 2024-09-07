from collections import OrderedDict
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import DATASET
from matplotlib.pyplot import text
from misc.helpers import append_and_create
from misc.plotting import set_seaborn_style

sys.dont_write_bytecode = True

from misc.config import Config
import ruptures as rpt

config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
print(done_workload_df.keys())


def _find_thresholds_and_plot_them(file_name, config):
    set_seaborn_style(font_size=10)
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_parquet(metric_file)
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

        average_improvement_over_all_time_steps = np.sum(current_row_np) / len(
            current_row_np
        )

        mean_from_poor_mens_plateau = np.mean(
            current_row_np[np.argmax(current_row_np > np.mean(current_row_np)) :]
        )

        window_size = 5

        for window_param in range(window_size, len(current_row_np)):
            # growing_window = current_row_np[-window_param:]
            fixed_window = current_row_np[
                (len(current_row_np) - window_param) : (
                    len(current_row_np) - window_param + window_size
                )
            ]

            # growing_average = np.mean(growing_window)

            fixed_average = np.mean(fixed_window)
            # print(f"{len(window)}: {window_std:.2f} {overall_stdt:.2f}")

            if fixed_average > average_improvement_over_all_time_steps:
                current_cutoff_values["fix"] = len(current_row_np) - window_param

            # if fixed_average > mean_from_poor_mens_plateau:
            #    current_cutoff_values["poor"] = len(current_row_np) - window_param

        if "fix" not in current_cutoff_values.keys():
            current_cutoff_values["fix"] = 1
        """beast_res = rb.beast(current_row_np, season="none", print_options=0)

        for cp_i, cp in enumerate(beast_res.trend.cp[:1]):
            if np.isnan(cp):
                continue
            current_cutoff_values[f"cp_{cp_i}"] = cp
        """
        # current_cutoff_values["mean"] = np.argmax(
        #    current_row_np > np.mean(current_row_np)
        # )
        """current_cutoff_values["poor"] = np.argmax(
            current_row_np
            > np.mean(
                current_row_np[np.argmax(current_row_np > np.mean(current_row_np)) :]
            )
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

        # ax.axhline(y=mean_from_poor_mens_plateau, lw=2, alpha=0.3, linestyle="--")
        ax.axhline(y=np.mean(current_row_np), lw=2, alpha=0.3, linestyle="--")

        ax.set_title(f"{EXP_DATASET.name} - {k}")

        ax.axvspan(0, current_cutoff_values["fix"], facecolor="red", alpha=0.2)
        ax.axvspan(
            current_cutoff_values["fix"],
            len(current_row_np),
            facecolor="blue",
            alpha=0.2,
        )

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


def _calculate_thresholds_and_save_them(file_name, config):
    metric_file = Path(file_name)
    df = pd.read_csv(metric_file).drop_duplicates()
    # print(df)

    small_done_df = done_workload_df.loc[
        done_workload_df["EXP_UNIQUE_ID"].isin(df.EXP_UNIQUE_ID)
    ].drop_duplicates()

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
    )["EXP_UNIQUE_ID"].apply(lambda rrr: rrr)

    """
    ich habe ein growing sliding window, und ab dem punkt wo das growing sliding window mehr "improvement" hat, als die durchschnittliche verbesserung -> da ist unser cutoff punkt
    also solange es kaum verbesserung im verlgeich zum durchschnitt gibt ->
    """
    for group, EXP_UNIQUE_ID in grouped.items():
        current_row = df.loc[df["EXP_UNIQUE_ID"] == EXP_UNIQUE_ID]
        del current_row["EXP_UNIQUE_ID"]
        current_row_np = current_row.to_numpy()[0]
        current_row_np = current_row_np[~np.isnan(current_row_np)]
        # print(current_row_np)

        average_improvement_over_all_time_steps = np.sum(np.diff(current_row_np)) / len(
            current_row_np
        )

        window_size = 10
        cutoff_value = None
        for window_param in range(window_size, len(current_row_np)):
            fixed_window = current_row_np[
                (len(current_row_np) - window_param) : (
                    len(current_row_np) - window_param + window_size
                )
            ]

            fixed_average = np.sum(np.diff(fixed_window)) / len(fixed_window)

            if fixed_average > average_improvement_over_all_time_steps:
                cutoff_value = len(current_row_np) - window_param + 1
                break

        if cutoff_value is None:
            cutoff_value = round(len(current_row_np) / 2)

        result = {kkk: vvv for kkk, vvv in zip(grouped.index.names, group)}
        del result[None]
        result["cutoff_value"] = cutoff_value
        result["EXP_DATASET"] = EXP_DATASET

        append_and_create(
            config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH,
            result,
        )


glob_list = [
    f
    for f in glob.glob(
        str(config.OUTPUT_PATH) + f"/ALIPY_RANDOM/*/accuracy.csv.xz",
        recursive=True,
    )
]

# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_calculate_thresholds_and_save_them)(file_name, config)
    for file_name in glob_list
)


# mergen done_workload_df and _dataset_dependent_random_ramp_plateau_threshold
old_plateau_df = pd.read_csv(
    config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH
)
done_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

del done_df["EXP_STRATEGY"]
del done_df["EXP_RANDOM_SEED"]
del done_df["EXP_NUM_QUERIES"]


merged = pd.merge(
    done_df,
    old_plateau_df,
    on=[kkk for kkk in old_plateau_df.columns if kkk != "cutoff_value"],
    how="outer",
)

merged = merged[["EXP_UNIQUE_ID", "cutoff_value"]]

merged.to_csv(config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH, index=False)
