import ast
from collections import defaultdict
import itertools
from pathlib import Path
import sys
import glob
from typing import Dict, List
import warnings
from matplotlib import pyplot as plt, use
from numpy import histogram_bin_edges
import pandas as pd
from tqdm import tqdm
from datasets import DATASET

from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing
import dask.dataframe as dd
import numpy as np
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, permutation_test

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


# oom_workload = dd.read_csv(str(config.OVERALL_STARTED_OOM_WORKLOAD_PATH))
done_workload = dd.read_csv(
    config.OVERALL_DONE_WORKLOAD_PATH,
    dtype={
        "EXP_BATCH_SIZE": "float64",
        "EXP_DATASET": "float64",
        "EXP_LEARNER_MODEL": "float64",
        "EXP_NUM_QUERIES": "float64",
        "EXP_RANDOM_SEED": "float64",
        "EXP_START_POINT": "float64",
        "EXP_STRATEGY": "float64",
        "EXP_TRAIN_TEST_BUCKET_SIZE": "float64",
        "EXP_UNIQUE_ID": "float64",
    },
)


# failed_workload = dd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)

# ich nehme mir eine kombination aus batch_size, learner_model, start_points, train_test_bucket, dataset
# ann sollten für diese kombination alle strategien ergebnisse haben

column_combinations = [
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    "EXP_NUM_QUERIES",
    "EXP_BATCH_SIZE",
    "EXP_LEARNER_MODEL",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
]


def _calculate_min_cutoffs():
    min_cutoffs = {}
    for cc in column_combinations:
        if cc in ["EXP_NUM_QUERIES", "EXP_RANDOM_SEED"]:
            continue
        print(cc)
        exp_ids_present_per_combination = (
            done_workload.groupby(by=[c for c in column_combinations if c != cc])[cc]
            .apply(len)
            .compute()
            .to_frame()
        )
        percentile_25 = np.percentile(exp_ids_present_per_combination[cc], 20)
        percentile_50 = np.percentile(exp_ids_present_per_combination[cc], 50)
        percentile_75 = np.percentile(exp_ids_present_per_combination[cc], 75)

        ax = sns.histplot(exp_ids_present_per_combination)
        ax.axvline(percentile_25)
        ax.axvline(percentile_50)
        ax.axvline(percentile_75)

        min_cutoffs[cc] = percentile_25
        plt.savefig(f"plots/{cc}.jpg", dpi=300)
        plt.clf()
    print(min_cutoffs)


def _get_dense_exp_ids(done_workload):
    if config.DENSE_WORKLOAD_PATH.exists():
        print(
            f"{config.DENSE_WORKLOAD_PATH} exists already, not caluclating again. Delete if this is unintended. And keep up the good work, you're doing a great job and look amazingly beautiful today!"
        )
        return

    for param in column_combinations:
        if param in ["EXP_NUM_QUERIES", "EXP_RANDOM_SEED"]:
            continue

        print(param)

        exp_ids_present_per_combination: pd.DataFrame = (
            done_workload.groupby(by=[c for c in column_combinations if c != param])[
                "EXP_UNIQUE_ID"
            ]
            .apply(list)
            .compute()
            .to_frame()
        )

        exp_ids_present_per_combination_lens = exp_ids_present_per_combination.apply(
            lambda x: len(x["EXP_UNIQUE_ID"]), axis=1
        )

        cutoff_value = np.percentile(exp_ids_present_per_combination_lens, 75)

        print(f"cutoff_value is {cutoff_value}")

        if cutoff_value == 1.0:
            continue

        # print(exp_ids_present_per_combination)
        exp_ids_present_per_combination = exp_ids_present_per_combination[
            exp_ids_present_per_combination.apply(
                lambda x: True if len(x["EXP_UNIQUE_ID"]) >= cutoff_value else False,
                axis=1,
            )
        ]

        exp_ids_merged = set(
            itertools.chain(*exp_ids_present_per_combination["EXP_UNIQUE_ID"].to_list())
        )

        before = len(done_workload)
        done_workload = done_workload.loc[
            done_workload["EXP_UNIQUE_ID"].isin(exp_ids_merged)
        ]
        print(f"reduced from {before} to {len(done_workload)}")

    done_workload.compute().to_csv(config.DENSE_WORKLOAD_PATH, index=False)


def _calculate_correlations(param_to_evaluate):
    dense_workload = pd.read_csv(
        config.DENSE_WORKLOAD_PATH,
    )

    dense_workload_grouped = dense_workload.groupby(
        by=[ddd for ddd in column_combinations if ddd != param_to_evaluate]
    ).apply(lambda r: list(zip(r[param_to_evaluate], r["EXP_UNIQUE_ID"])))

    print(
        f"Calculating correlations for {param_to_evaluate}: {len(dense_workload_grouped)}"
    )

    combined_stats = []

    pbar = tqdm(
        dense_workload_grouped.reset_index().iterrows(),
        total=dense_workload_grouped.shape[0],
    )
    for _, row in pbar:
        path_glob = f"{config.OUTPUT_PATH}/{AL_STRATEGY(row.EXP_STRATEGY).name}/{DATASET(row.EXP_DATASET).name}/*.csv.xz"

        pbar.set_description(path_glob)

        metrics_data = defaultdict(lambda: defaultdict(int))
        for metric_path in glob.glob(
            path_glob,
            recursive=True,
        ):
            metrics_not_suitable_for_comparisons = [
                "selected_indices",
                "y_pred_test",
                "y_pred_train",
                "query_selection_time",
                "learner_training_time",
            ]

            ignore_metric = False
            for mnsfc in metrics_not_suitable_for_comparisons:
                if metric_path.endswith(f"{mnsfc}.csv.xz"):
                    ignore_metric = True
                    continue
            if ignore_metric:
                continue

            # ignore cut-point/auc metrics because they are simply summed/averaged over the other metrics -> the correlations will be found in the original metrics, not in them!
            if "auc_" in metric_path:
                continue
            if "learning_stability_" in metric_path:
                continue

            metric_path = Path(metric_path)
            # print(metric_path)

            metric_df = pd.read_csv(metric_path)

            if metric_df.shape[1] < 3:
                print("Metric has too few columns, exiting.")
                print(metric_path)
                exit(-1)

            for param_to_evaluate_value, EXP_UNIQUE_ID in row[0]:
                # print(param_to_evaluate_value)
                value = (
                    metric_df.loc[metric_df["EXP_UNIQUE_ID"] == EXP_UNIQUE_ID]
                    .drop("EXP_UNIQUE_ID", axis=1)
                    .iloc[0]
                    .to_list()
                )
                metric_name = str(metric_path.name).removesuffix(".csv.xz")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    value = [
                        np.nanmean(ast.literal_eval(str(vvv)))
                        if str(vvv) != "nan"
                        else np.nan
                        for vvv in value
                    ]
                metrics_data[param_to_evaluate_value][
                    metric_name
                ] = value  # np.nanmean(value)

        # print(metrics_data)
        metrics_df = pd.DataFrame(metrics_data)
        # print(metrics_df)

        # now i calculate the correlation PER metric pair!

        metrics_df["spearmanr_stat"], metrics_df["spearmanr_pvalue"] = zip(
            *metrics_df.apply(
                lambda rrr: spearmanr(rrr.iloc[0], rrr.iloc[1], nan_policy="omit"),
                axis=1,
            )
        )

        # print(metrics_df)

        combined_stats.append(
            [metrics_df["spearmanr_stat"], metrics_df["spearmanr_pvalue"]]
        )

    print(combined_stats)
    exit(-1)


_get_dense_exp_ids(done_workload)

for cc in [
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_STRATEGY",
    # "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    # "EXP_NUM_QUERIES",
    "EXP_LEARNER_MODEL",
    "EXP_TRAIN_TEST_BUCKET_SIZE" "METRICS",
]:
    _calculate_correlations(cc)
# _get_dense_exp_ids(done_workload)
exit(-1)

minimal_set = None
for row in exp_ids_present_per_dataset:
    if minimal_set is None:
        minimal_set = row
    else:
        minimal_set = minimal_set.intersection(row)

print(minimal_set)

exit(-1)
exp_ids_present_per_strategy = (
    done_workload.groupby(by=[c for c in column_combinations if c != "EXP_STRATEGY"])[
        "EXP_STRATEGY"
    ]
    .apply(set)
    .compute()
)
print(exp_ids_present_per_strategy)
