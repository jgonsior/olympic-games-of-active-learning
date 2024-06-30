import csv
import multiprocessing
import subprocess
import sys
from typing import Dict
import numpy as np
import pandas as pd
from pathlib import Path

import scipy
from sklearn.preprocessing import RobustScaler

from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

default_standard_metric = "full_auc_weighted_f1-score"
grid_type = "sparse"
rank_or_percentage = "dataset_normalized_percentages"
interpolation = "average_of_same_strategy"

hyperparameters_to_evaluate = [
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_START_POINT",
    "standard_metric",
    "auc_metric",
]


def read_or_create_ts(metric_name) -> pd.DataFrame:
    if not Path(config.CORRELATION_TS_PATH / f"{metric_name}.parquet").exists():
        unsorted_f = config.CORRELATION_TS_PATH / f"{metric_name}.unsorted.csv"
        unparqueted_f = config.CORRELATION_TS_PATH / f"{metric_name}.to_parquet.csv"

        if not unsorted_f.exists() and not unparqueted_f.exists():
            log_and_time("Create selected indices ts")
            create_fingerprint_joined_timeseries_csv_files(
                metric_names=[metric_name], config=config
            )

        if not unparqueted_f.exists():
            log_and_time("Created, now sorting")
            command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {unsorted_f} -o {config.CORRELATION_TS_PATH}/{metric_name}.to_parquet.csv"
            print(command)
            subprocess.run(command, shell=True, text=True)
            unsorted_f.unlink()

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
        ts["metric_value"] = ts["metric_value"].apply(
            lambda xxx: (
                np.fromstring(
                    str(xxx).removeprefix("[").removesuffix("]"),
                    dtype=np.int32,
                    sep=",",
                )
            )
        )

        f = Path(config.CORRELATION_TS_PATH / f"{metric_name}.parquet")
        ts.to_parquet(f)
        unparqueted_f.unlink()

    ts = pd.read_parquet(
        config.CORRELATION_TS_PATH / f"{metric_name}.parquet",
        columns=[
            "EXP_DATASET",
            "EXP_STRATEGY",
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
            # "ix",
            # "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )
    return ts


for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
    ranking_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
    )
    ranking_path.parent.mkdir(parents=True, exist_ok=True)

    if ranking_path.exists():
        print(f"{ranking_path} already exists")
        continue

    if hyperparameter_to_evaluate == "standard_metric":
        hyperparameter_values = [
            f"full_auc_{kkk}"
            for kkk in [
                "accuracy",
                "weighted_recall",
                "macro_f1-score",
                "macro_precision",
                "macro_recall",
                "weighted_f1-score",
                "weighted_precision",
            ]
        ]
    elif hyperparameter_to_evaluate == "auc_metric":
        hyperparameter_values = [
            f"{kkk}_weighted_f1-score"
            for kkk in [
                "final_value_",
                "first_5_",
                "full_auc_",
                "last_5_",
                "learning_stability_5_",
                "learning_stability_10_",
                "ramp_up_auc_",
                "plateau_auc_",
            ]
        ]

    ranking_dict: Dict[str, np.ndarray] = {}

    if hyperparameter_to_evaluate not in ["standard_metric", "auc_metric"]:
        ts = read_or_create_ts(default_standard_metric)

        ts_orig = ts.copy()
        hyperparameter_values = ts[hyperparameter_to_evaluate].unique()

    for hyperparameter_target_value in hyperparameter_values:
        if hyperparameter_to_evaluate in ["standard_metric", "auc_metric"]:
            standard_metric = hyperparameter_to_evaluate
            ts = read_or_create_ts(hyperparameter_target_value)
        else:
            ts = ts_orig.copy()
            standard_metric = default_standard_metric

        if hyperparameter_to_evaluate not in ["auc_metric", "standard_metric"]:
            ts = ts.loc[ts[hyperparameter_to_evaluate] == hyperparameter_target_value]

        fingerprint_cols = list(ts.columns)
        fingerprint_cols.remove("metric_value")
        fingerprint_cols.remove("EXP_DATASET")
        fingerprint_cols.remove("EXP_STRATEGY")

        ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
            lambda row: "_".join([str(rrr) for rrr in row]), axis=1
        )

        ts["dataset_strategy"] = ts[["EXP_DATASET", "EXP_STRATEGY"]].parallel_apply(
            lambda row: "_".join([str(rrr) for rrr in row]), axis=1
        )

        for fg_col in fingerprint_cols:
            del ts[fg_col]

        log_and_time("Done fingerprinting")
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
            with open(shared_fingerprints_csv_path, "w") as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(list(shared_fingerprints))
                wr.writerow([amount_of_max_shared_fingerprints])

        if grid_type == "dense":
            ts = ts.loc[(ts["fingerprint"].isin(shared_fingerprints))]

        del ts["dataset_strategy"]
        del ts["fingerprint"]

        ts = (
            ts.groupby(by=["EXP_DATASET", "EXP_STRATEGY"])["metric_value"]
            .apply(lambda lll: np.array([llllll for llllll in lll]).flatten())
            .reset_index()
        )
        ts = ts.pivot(
            index="EXP_DATASET", columns="EXP_STRATEGY", values="metric_value"
        )

        if rank_or_percentage == "dataset_normalized_percentages":

            def _flatten(xss):
                return [[x] for xs in xss for x in xs]

            def _unflatten(xss):
                return [xs[0] for xs in xss]

            def _dataset_normalized_percentages(row: pd.Series) -> pd.Series:
                row = row.dropna()
                transformer = RobustScaler().fit(
                    _flatten([rrr.tolist() for rrr in row.to_list()])
                )
                data = [[[rxrxrx] for rxrxrx in rrr] for rrr in row]
                result = [transformer.transform(rrr) for rrr in data]

                result = pd.Series([_unflatten(rrr) for rrr in result], index=row.index)
                return result

            # ts = ts.parallel_apply(_dataset_normalized_percentages, axis=1)
            ts = ts.parallel_apply(_dataset_normalized_percentages, axis=1)

        if grid_type == "sparse":
            # remove combinations which are not sparse
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

            match interpolation:
                case "remove":
                    ts = ts.parallel_applymap(_remove_sparse)
                case "zero":
                    ts = ts.parallel_applymap(_zero_interpolation)
                case "average_of_same_strategy":
                    ts = ts.parallel_applymap(_average_of_same_strategy_interpolation)
        ts = ts.parallel_applymap(np.mean)

        if rank_or_percentage == "rank":

            def _calculate_ranks(row: pd.Series) -> pd.Series:
                ranks = scipy.stats.rankdata(row, method="max", nan_policy="omit")

                # amount_of_non_nan_values = np.count_nonzero(~np.isnan(ranks))
                result = pd.Series(ranks, index=row.index)
                return result

            ts = ts.parallel_apply(_calculate_ranks, axis=1)

        ts.columns = [AL_STRATEGY(int(kkk)).name for kkk in ts.columns]

        ts = ts.set_index([[DATASET(int(kkk)).name for kkk in ts.index]])

        ts = ts.T
        ts.loc[:, "Total"] = ts.mean(axis=1)

        if rank_or_percentage == "rank":
            ts.sort_values(by=["Total"], inplace=True, ascending=True)
        else:
            ts.sort_values(by=["Total"], inplace=True, ascending=True)
        ts = ts.T
        print(ts)

        ranking_dict[f"{hyperparameter_to_evaluate}: {hyperparameter_target_value}"] = (
            ts.loc["Total"]
        )

    ranking_df = pd.DataFrame(ranking_dict).T
    ranking_df.to_csv(ranking_path)
