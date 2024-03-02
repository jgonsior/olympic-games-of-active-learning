import copy
import itertools
from pathlib import Path
import sys
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing

from misc.helpers import append_and_create, get_df, get_glob_list


sys.dont_write_bytecode = True

from misc.config import Config


config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)


def _is_standard_metric(metric_path: Path) -> bool:
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
        if f"{sm}.csv.xz" == metric_path.name:
            return True
        elif f"{sm}.csv.xz.parquet" == metric_path.name:
            return True

    return False


def _do_stuff(exp_dataset, exp_strategy, config):
    glob_list = get_glob_list(
        config, limit=f"/{exp_strategy.name}/{exp_dataset.name}/*"
    )

    if len(glob_list) == 0:
        return

    metric_dfs = {}
    exp_ids_per_metric = {}

    exp_ids_union = set()
    for file_name in glob_list:
        print(file_name)
        if not _is_standard_metric(file_name):
            print(f"Not standard metric: {file_name}")

        if file_name.name.endswith(".parquet"):
            metric_name = file_name.name.removesuffix(".csv.xz.parquet")
        else:
            metric_name = file_name.name.removesuffix(".csv.xz")

        metric_df = get_df(file_name, config)

        if metric_df is None:
            return

        metric_dfs[metric_name] = metric_df

        exp_ids_per_metric[metric_name] = set(
            metric_dfs[metric_name]["EXP_UNIQUE_ID"].to_list()
        )

        exp_ids_union = exp_ids_union.union(exp_ids_per_metric[metric_name])

    exp_ids_intersection = copy.deepcopy(exp_ids_union)
    print(len(exp_ids_union))

    for _, exp_ids in exp_ids_per_metric.items():
        exp_ids_intersection = exp_ids_intersection.intersection(exp_ids)
        print(len(exp_ids))
    print(len(exp_ids_intersection))
    exit(-1)

    difference = exp_ids_union.symmetric_difference(exp_ids_intersection)
    if len(difference) > 0:
        for broken_exp_id in difference:
            potential_rows_to_be_deleted = done_workload_df.loc[
                done_workload_df["EXP_UNIQUE_ID"] == broken_exp_id
            ]

            if len(potential_rows_to_be_deleted) > 0:
                row_to_be_deleted = potential_rows_to_be_deleted.iloc[0]
                append_and_create(
                    config.MISSING_EXP_IDS_IN_METRIC_FILES,
                    row_to_be_deleted.to_dict(),
                )


Parallel(n_jobs=1, verbose=10)(
    #  Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(exp_dataset, exp_strategy, config)
    for (exp_dataset, exp_strategy) in itertools.product(
        config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY
    )
)
