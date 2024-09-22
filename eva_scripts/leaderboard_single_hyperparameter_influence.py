import multiprocessing
import random
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
)
from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=False, use_memory_fs=False
)

default_standard_metric = "full_auc_weighted_f1-score"
grid_type = "sparse"  # dense is not supported by this script!
rank_or_percentage = "dataset_normalized_percentages"
interpolation = "average_of_same_strategy"

hyperparameters_to_evaluate = [
    "standard_metric",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_START_POINT",
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
            f"{kkk}weighted_f1-score"
            for kkk in [
                "final_value_",
                "first_5_",
                "full_auc_",
                "last_5_",
                # "learning_stability_5_",
                # "learning_stability_10_",
                "ramp_up_auc_",
                "plateau_auc_",
            ]
        ]
    elif hyperparameter_to_evaluate == "start_point_scenario":
        hyperparameter_values = list(
            enumerate(
                [
                    20,
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                    *list(range(1, 20)),
                ]
            )
        )
    elif hyperparameter_to_evaluate == "dataset_scenario":
        hyperparameter_values = list(
            enumerate(
                [
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                    *list(range(1, 100)),
                ]
            )
        )

    ranking_dict: Dict[str, np.ndarray] = {}

    if hyperparameter_to_evaluate not in [
        "standard_metric",
        "auc_metric",
        "start_point_scenario",
        "dataset_scenario",
    ]:
        ts = read_or_create_ts(default_standard_metric)
        ts_orig = ts.copy()
        hyperparameter_values = ts[hyperparameter_to_evaluate].unique()
    elif hyperparameter_to_evaluate in ["start_point_scenario", "dataset_scenario"]:
        ts = read_or_create_ts(default_standard_metric)
        ts_orig = ts.copy()

    for hyperparameter_target_value in hyperparameter_values:
        if hyperparameter_to_evaluate in ["standard_metric", "auc_metric"]:
            ts = read_or_create_ts(hyperparameter_target_value)
        else:
            ts = ts_orig.copy()

        if hyperparameter_to_evaluate not in [
            "auc_metric",
            "standard_metric",
            "start_point_scenario",
            "dataset_scenario",
        ]:
            ts = ts.loc[ts[hyperparameter_to_evaluate] == hyperparameter_target_value]
        elif hyperparameter_to_evaluate == "start_point_scenario":
            if hyperparameter_target_value[1] > len(config.EXP_GRID_START_POINT):
                continue
            allowed_start_points = random.sample(
                config.EXP_GRID_START_POINT, hyperparameter_target_value[1]
            )

            ts = ts.loc[ts["EXP_START_POINT"].isin(allowed_start_points)]
        elif hyperparameter_to_evaluate == "dataset_scenario":
            if hyperparameter_target_value[1] > len(config.EXP_GRID_DATASET):
                continue
            allowed_start_points = [
                kkk
                for kkk in random.sample(
                    config.EXP_GRID_DATASET, hyperparameter_target_value[1]
                )
            ]
            ts = ts.loc[ts["EXP_DATASET"].isin(allowed_start_points)]

        ts = (
            ts.groupby(by=["EXP_DATASET", "EXP_STRATEGY"])["metric_value"]
            .apply(lambda lll: np.array([llllll for llllll in lll]).flatten())
            .reset_index()
        )
        ts = ts.pivot(
            index="EXP_DATASET", columns="EXP_STRATEGY", values="metric_value"
        )

        amount_of_max_shared_fingerprints = ts.parallel_applymap(np.shape).max(
            axis=None
        )
        print("important comment for parallel voodo reasons")
        amount_of_max_shared_fingerprints = amount_of_max_shared_fingerprints[0]

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
        print(ts)
        if rank_or_percentage == "dataset_normalized_percentages":

            def _flatten(xss):
                return [[x] for xs in xss for x in xs]

            def _unflatten(xss):
                return [xs[0] for xs in xss]

            def _dataset_normalized_percentages(row: pd.Series) -> pd.Series:
                row = row.dropna()
                transformer = RobustScaler().fit(
                    row  # _flatten([rrr for rrr in row.to_list()])
                )
                data = [[[rxrxrx] for rxrxrx in rrr] for rrr in row]
                result = [transformer.transform(rrr) for rrr in data]

                result = pd.Series([_unflatten(rrr) for rrr in result], index=row.index)
                return result

            # ts = ts.parallel_apply(_dataset_normalized_percentages, axis=1)
            ts = ts.parallel_apply(_dataset_normalized_percentages, axis=1)

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
