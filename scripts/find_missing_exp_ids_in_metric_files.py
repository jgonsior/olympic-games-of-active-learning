import copy
import csv
import itertools
from pathlib import Path
import sys
import glob
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing


sys.dont_write_bytecode = True

from misc.config import Config


config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)


def _is_standard_metric(metric_path: str) -> bool:
    standard_metrics = [
        "accuracy",
        "weighted_recall",
        "macro_f1-score",
        "macro_precision",
        "macro_recall",
        "weighted_f1-score",
        "weighted_precision",
        "weighted_recall",
        "learner_training_time",
        "query_selection_time",
        "y_pred_test",
        "y_pred_train",
        "selected_indices",
    ]

    for sm in standard_metrics:
        if f"/{sm}.csv" in metric_path:
            return True
    return False


def _do_stuff(exp_dataset, exp_strategy, config):
    if exp_strategy.name != "SKACTIVEML_DWUS":
        return
    glob_list = [
        *[
            f
            for f in glob.glob(
                str(config.OUTPUT_PATH)
                + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv.xz",
                recursive=True,
            )
        ],
        *[
            f
            for f in glob.glob(
                str(config.OUTPUT_PATH)
                + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv.xz.parquet",
                recursive=True,
            )
        ],
    ]

    if len(glob_list) == 0:
        return

    metric_dfs = {}
    exp_ids_per_metric = {}

    exp_ids_union = set()
    for file_name in glob_list:
        if not _is_standard_metric(file_name):
            print(f"Not standard metric: {file_name}")
        metric_name = Path(file_name).name.removesuffix(".csv.xz")
        metric_name = Path(file_name).name.removesuffix(".csv.xz.parquet")

        if file_name.endswith(".csv.xz"):
            metric_dfs[metric_name] = pd.read_csv(file_name)
        else:
            metric_dfs[metric_name] = pd.read_parquet(file_name)
        exp_ids_per_metric[metric_name] = set(
            metric_dfs[metric_name]["EXP_UNIQUE_ID"].to_list()
        )

        exp_ids_union = exp_ids_union.union(exp_ids_per_metric[metric_name])
    exp_ids_intersection = copy.deepcopy(exp_ids_union)

    for metric, exp_ids in exp_ids_per_metric.items():
        exp_ids_intersection = exp_ids_intersection.intersection(exp_ids)

    if len(exp_ids_intersection) < len(exp_ids_union):
        if not config.MISSING_EXP_IDS_IN_METRIC_FILES.exists():
            with open(config.MISSING_EXP_IDS_IN_METRIC_FILES, "a") as f:
                w = csv.DictWriter(f, fieldnames=done_workload_df.keys())
                w.writeheader()

        with open(config.MISSING_EXP_IDS_IN_METRIC_FILES, "a") as f:
            w = csv.DictWriter(f, fieldnames=done_workload_df.keys())

            for broken_exp_id in exp_ids_intersection:
                w.writerow(
                    done_workload_df.loc[
                        done_workload_df["EXP_UNIQUE_ID"] == broken_exp_id
                    ]
                    .to_numpy()
                    .tolist()
                )
        return
        for metric, exp_ids in exp_ids_per_metric.items():
            if exp_ids_per_metric[metric].difference(exp_ids_intersection) > 0:
                for file_name in glob_list:
                    if file_name.endswith(".csv.xz"):
                        metric_df = pd.read_csv(file_name)
                    else:
                        metric_df = pd.read_parquet(file_name)

                    metric_df = metric_df.loc[
                        metric_df["EXP_UNIQUE_ID"].isin(exp_ids_intersection)
                    ]

                    if file_name.endswith(".csv.xz"):
                        metric_df.to_csv(file_name, index=False)
                    else:
                        metric_df.to_parquet(file_name)

        # we've got a problem

    # find missing exp_ids as difference among all set

    # remove those from all metric_dfs where they ARE existent

    # rerun those experiments, merge .csv.xz and .csv


Parallel(n_jobs=1, verbose=10)(
    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(exp_dataset, exp_strategy, config)
    for (exp_dataset, exp_strategy) in itertools.product(
        config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY
    )
)
