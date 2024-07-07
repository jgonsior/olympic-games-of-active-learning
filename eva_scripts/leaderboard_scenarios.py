import multiprocessing
import random
import subprocess
import sys

sys.dont_write_bytecode = True


from datasets import DATASET
from resources.data_types import AL_STRATEGY, LEARNER_MODEL
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import scipy
from datasets import DATASET
from misc.helpers import (
    append_and_create,
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
)
from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

prepare_eva_pathes(config.SCENARIOS, config)

default_standard_metric = "full_auc_weighted_f1-score"


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


ts = read_or_create_ts(default_standard_metric)
ts_orig = ts.copy()

if config.EVA_MODE == "create":

    def flatten(xss):
        return [x for xs in xss for x in xs]

    if config.SCENARIOS == "dataset_scenario":
        hyperparameter_values = list(
            # enumerate(flatten([list(range(1, 2)) for _ in range(0, 400)]))
            enumerate(flatten([list(range(1, 92)) for _ in range(0, 300)]))
        )
    elif config.SCENARIOS == "start_point_scenario":
        hyperparameter_values = list(
            # enumerate([20, *flatten([list(range(1, 20)) for _ in range(0, 4)])])
            enumerate([21, *flatten([list(range(1, 21)) for _ in range(0, 1500)])])
        )
    elif config.SCENARIOS == "adv_start_scenario":
        hyperparameter_values = list(
            # enumerate([20, *flatten([list(range(1, 20)) for _ in range(0, 4)])])
            enumerate([21, *flatten([list(range(1, 21)) for _ in range(0, 1500)])])
        )

    create_workload(
        hyperparameter_values,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1,
        SCRIPTS_PATH="metrics",
        SLURM_NR_THREADS=1,
        script_type="metrics",
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:
    default_standard_metric = "full_auc_weighted_f1-score"
    grid_type = "sparse"  # dense is not supported by this script!
    rank_or_percentage = "dataset_normalized_percentages"
    interpolation = "average_of_same_strategy"

    def _run_single_metric(ix, hyperparameter_target_value, config: Config):
        random.seed(ix)
        hyperparameter_target_value = (ix, hyperparameter_target_value)
        ts = ts_orig.copy()
        if config.SCENARIOS == "start_point_scenario":
            if hyperparameter_target_value[1] > len(config.EXP_GRID_START_POINT):
                return

            allowed_start_points = random.sample(
                config.EXP_GRID_START_POINT, hyperparameter_target_value[1]
            )

            ts = ts.loc[ts["EXP_START_POINT"].isin(allowed_start_points)]
        elif config.SCENARIOS == "adv_start_scenario":
            if hyperparameter_target_value[1] > len(config.EXP_GRID_START_POINT):
                return

            allowed_start_points = random.sample(
                config.EXP_GRID_START_POINT, hyperparameter_target_value[1]
            )

            # limit to less other parameters
            ts = ts.loc[
                (ts["EXP_LEARNER_MODEL"] == LEARNER_MODEL.RF)
                & (ts["EXP_BATCH_SIZE"] == 20)
            ]
            ts = ts.loc[ts["EXP_START_POINT"].isin(allowed_start_points)]

        elif config.SCENARIOS == "dataset_scenario":
            if hyperparameter_target_value[1] > len(config.EXP_GRID_DATASET):
                return
            allowed_start_points = [
                kkk
                for kkk in random.sample(
                    ts["EXP_DATASET"].unique().tolist(), hyperparameter_target_value[1]
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
            ts = ts.apply(_dataset_normalized_percentages, axis=1)
        amount_of_max_shared_fingerprints = ts.map(np.shape).max(axis=None)
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
                    ts = ts.map(_remove_sparse)
                case "zero":
                    ts = ts.map(_zero_interpolation)
                case "average_of_same_strategy":
                    ts = ts.map(_average_of_same_strategy_interpolation)
        ts = ts.map(np.mean)

        if rank_or_percentage == "rank":

            def _calculate_ranks(row: pd.Series) -> pd.Series:
                ranks = scipy.stats.rankdata(row, method="max", nan_policy="omit")

                # amount_of_non_nan_values = np.count_nonzero(~np.isnan(ranks))
                result = pd.Series(ranks, index=row.index)
                return result

            ts = ts.apply(_calculate_ranks, axis=1)

        ts.columns = [AL_STRATEGY(int(kkk)).name for kkk in ts.columns]

        ts = ts.set_index([[DATASET(int(kkk)).name for kkk in ts.index]])

        ts = ts.T
        ts.loc[:, "Total"] = ts.mean(axis=1)

        if rank_or_percentage == "rank":
            ts.sort_values(by=["Total"], inplace=True, ascending=True)
        else:
            ts.sort_values(by=["Total"], inplace=True, ascending=True)
        ts = ts.T

        append_and_create(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{config.SCENARIOS}.csv",
            {
                "": f"{config.SCENARIOS}: {hyperparameter_target_value}",
                **(dict(sorted(ts.loc["Total"].to_dict().items()))),
            },
        )

    run_from_workload(do_stuff=_run_single_metric, config=config)
