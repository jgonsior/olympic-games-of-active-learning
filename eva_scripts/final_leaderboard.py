import csv
import multiprocessing
import subprocess
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.style as mplstyle

import matplotlib as mpl
import scipy
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

orig_standard_metric = "weighted_f1-score"


# what are the different ways for leaderboard?
# DONE aggregation over aggregation
# ranking only (spearman instead of pearson)
# normalize datasets first

# 1) only keep complete grid, remove all strategy/dataset combinations which are not 960 elements long
# 2) interpolate with 0
# 3) interpolate with average result of this strategy for this dataset?
# 4) interpolate with average rank this strategy got for other datasets?

interpolation_strategies = [
    "count",
    "count_dense",
    "remove",
    "zero",
    "average_of_same_strategy",
    "average_rank",
]

for rank_or_percentage in ["rank", "percentages"]:
    for grid_type in ["sparse", "dense"]:
        for interpolation in interpolation_strategies:
            if grid_type == "dense" and interpolation != "zero":
                continue

            for auc_prefix in [
                "full_auc_",
                # "first_5_",
                # "last_5_",
                # "ramp_up_auc_",
                # "plateau_auc_",
                # "final_value_",
            ]:
                standard_metric = auc_prefix + orig_standard_metric

                log_and_time(f"Calculating for {standard_metric}")

                if not Path(
                    config.CORRELATION_TS_PATH / f"{standard_metric}.parquet"
                ).exists():
                    unsorted_f = (
                        config.CORRELATION_TS_PATH / f"{standard_metric}.unsorted.csv"
                    )
                    unparqueted_f = (
                        config.CORRELATION_TS_PATH / f"{standard_metric}.to_parquet.csv"
                    )

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
                    """ts["metric_value"] = ts["metric_value"].apply(
                        lambda xxx: (
                            np.fromstring(
                                xxx.removeprefix("[").removesuffix("]"),
                                dtype=np.int32,
                                sep=" ",
                            )
                        )
                    )"""

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
                        # "ix",
                        # "EXP_UNIQUE_ID_ix",
                        "metric_value",
                    ],
                )
                print(f"{standard_metric}.parquet")
                print(ts)

                fingerprint_cols = list(ts.columns)
                fingerprint_cols.remove("metric_value")
                fingerprint_cols.remove("EXP_DATASET")
                fingerprint_cols.remove("EXP_STRATEGY")

                ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
                    lambda row: "_".join([str(rrr) for rrr in row]), axis=1
                )

                ts["dataset_strategy"] = ts[
                    ["EXP_DATASET", "EXP_STRATEGY"]
                ].parallel_apply(
                    lambda row: "_".join([str(rrr) for rrr in row]), axis=1
                )

                for fg_col in fingerprint_cols:
                    del ts[fg_col]

                log_and_time("Done fingerprinting")
                print(ts)
                # exit(-1)

                shared_fingerprints_csv_path = (
                    config.CORRELATION_TS_PATH
                    / f"final_leaderboard_shared_fingerprints_{standard_metric}.csv"
                )
                if shared_fingerprints_csv_path.exists():
                    with open(shared_fingerprints_csv_path, newline="") as f:
                        reader = list(csv.reader(f))
                        shared_fingerprints = set(reader[0])
                        amount_of_max_shared_fingerprints = int(reader[1][0])
                else:
                    shared_fingerprints = None
                    amount_of_max_shared_fingerprints = 0
                    for target_value in ts["dataset_strategy"].unique():
                        tmp_fingerprints = set(
                            ts.loc[ts["dataset_strategy"] == target_value][
                                "fingerprint"
                            ].to_list()
                        )

                        if len(tmp_fingerprints) > amount_of_max_shared_fingerprints:
                            amount_of_max_shared_fingerprints = len(tmp_fingerprints)

                        if shared_fingerprints is None:
                            print(target_value)
                            shared_fingerprints = tmp_fingerprints
                        else:
                            print(f"{target_value}: {len(shared_fingerprints)}")
                            shared_fingerprints = shared_fingerprints.intersection(
                                tmp_fingerprints
                            )

                    log_and_time(
                        f"Done calculating shared fingerprints - {len(shared_fingerprints)} - #{amount_of_max_shared_fingerprints}"
                    )
                    print(shared_fingerprints)
                    with open(shared_fingerprints_csv_path, "w") as myfile:
                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                        wr.writerow(list(shared_fingerprints))
                        wr.writerow([amount_of_max_shared_fingerprints])

                if grid_type == "dense" or interpolation == "count_dense":
                    ts = ts.loc[(ts["fingerprint"].isin(shared_fingerprints))]

                print(ts)
                del ts["dataset_strategy"]
                del ts["fingerprint"]

                # @todo shared fingerprints hier betrachten!
                # was mache ich mit lücken? z. B. quire :/
                # lücken wegen error -> alles weg?
                # lücken wegen timeout -> 0%? oder so viel wie random bei iteration 0 hat?
                # aktuell ignoriere ich lücken einfach???
                # interpolation -> (90-len(df))*interpolation value in der mittelwert berechnung mit hinzu nehmen

                ts = (
                    ts.groupby(by=["EXP_DATASET", "EXP_STRATEGY"])["metric_value"]
                    .apply(lambda lll: np.array([llllll for llllll in lll]).flatten())
                    .reset_index()
                )
                ts = ts.pivot(
                    index="EXP_DATASET", columns="EXP_STRATEGY", values="metric_value"
                )
                print(ts)

                if grid_type == "sparse":
                    # remove combinations which are not sparse
                    def _count_sparse(cell):
                        if type(cell) == float:
                            return cell
                        if len(cell) < amount_of_max_shared_fingerprints:
                            return len(cell)
                        else:
                            return len(cell)

                    def _remove_sparse(cell):
                        if type(cell) == float:
                            return cell
                        if len(cell) < amount_of_max_shared_fingerprints:
                            return []
                        else:
                            return cell

                    def _zero_interpolation(cell):
                        if type(cell) == float:
                            return [0]
                        if len(cell) < amount_of_max_shared_fingerprints:
                            return [
                                *cell,
                                *[
                                    0
                                    for _ in range(
                                        0, amount_of_max_shared_fingerprints - len(cell)
                                    )
                                ],
                            ]
                        else:
                            return cell

                    def _average_of_same_strategy_interpolation(cell):
                        average_of_this_strategy = np.mean(cell)
                        if type(cell) == float:
                            return [average_of_this_strategy]
                        if len(cell) < amount_of_max_shared_fingerprints:
                            return [
                                *cell,
                                *[
                                    average_of_this_strategy
                                    for _ in range(
                                        0, amount_of_max_shared_fingerprints - len(cell)
                                    )
                                ],
                            ]
                        else:
                            return cell

                    def _average_rank_interpolation(cell):
                        if type(cell) == float:
                            return [0]
                        if len(cell) < amount_of_max_shared_fingerprints:
                            return [
                                *cell,
                                *[
                                    0
                                    for _ in range(
                                        0, amount_of_max_shared_fingerprints - len(cell)
                                    )
                                ],
                            ]
                        else:
                            return cell

                    match interpolation:
                        case "count":
                            ts = ts.parallel_applymap(_count_sparse)
                        case "count_dense":
                            ts = ts.parallel_applymap(_count_sparse)
                        case "remove":
                            ts = ts.parallel_applymap(_remove_sparse)
                        case "zero":
                            ts = ts.parallel_applymap(_zero_interpolation)
                        case "average_of_same_strategy":
                            ts = ts.parallel_applymap(
                                _average_of_same_strategy_interpolation
                            )
                        case "average_rank":
                            ts = ts.parallel_applymap(_average_rank_interpolation)

                ts = ts.parallel_applymap(np.mean)
                # print(ts)

                if rank_or_percentage == "rank":
                    columns = ts.columns

                    def _calculate_ranks(row: pd.Series) -> pd.Series:
                        # print(row)
                        ranks = scipy.stats.rankdata(row, method="max")
                        result = pd.Series(len(row) - ranks + 1, index=row.index)
                        return result

                    ts = ts.parallel_apply(_calculate_ranks, axis=1)
                    # print(columns)
                    # print(ts.columns)
                    # ts.columns = columns
                # ts = ts.parallel_applymap(np.median)
                # print(ts)

                ts.columns = [AL_STRATEGY(int(kkk)).name for kkk in ts.columns]

                destination_path = Path(
                    config.OUTPUT_PATH
                    / f"plots/final_leaderboard/{rank_or_percentage}_{grid_type}_{interpolation}_{standard_metric}"
                )
                destination_path.parent.mkdir(exist_ok=True, parents=True)

                ts = ts.set_index([[DATASET(int(kkk)).name for kkk in ts.index]])

                ts = ts.T
                ts.loc[:, "Total"] = ts.mean(axis=1)
                ts.sort_values(by=["Total"], inplace=True)
                ts = ts.T
                print(ts)

                print(str(destination_path) + f".jpg")
                set_seaborn_style(font_size=8)
                mpl.rcParams["path.simplify"] = True
                mpl.rcParams["path.simplify_threshold"] = 1.0
                # plt.figure(figsize=set_matplotlib_size(fraction=10))

                # calculate fraction based on length of keys
                plt.figure(figsize=set_matplotlib_size(fraction=len(ts.columns) / 6))

                if (
                    rank_or_percentage == "rank"
                    or interpolation == "count"
                    or interpolation == "count_dense"
                ):
                    ax = sns.heatmap(ts, annot=True)
                else:
                    ax = sns.heatmap(ts, annot=True, fmt=".2%")

                ax.set_title(
                    f"Final leaderboard: {rank_or_percentage} {grid_type} {interpolation} {standard_metric}"
                )

                ts.to_parquet(str(destination_path) + f".parquet")

                plt.savefig(
                    str(destination_path) + f".jpg",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0,
                )
                # goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
