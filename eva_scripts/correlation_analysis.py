import itertools
from pathlib import Path
import sys
import glob
from typing import Dict, List
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

from scipy.stats import pearsonr

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


def _calculate_correlations():
    dense_workload = pd.read_csv(
        config.DENSE_WORKLOAD_PATH,
    )

    def _do_stuff_parallel(row):
        print(
            f"{config.OUTPUT_PATH}/{AL_STRATEGY(row.EXP_STRATEGY).name}/{DATASET(row.EXP_DATASET).name}/*.csv.xz"
        )

        metrics_data = []
        for metric_path in glob.glob(
            f"{config.OUTPUT_PATH}/{AL_STRATEGY(row.EXP_STRATEGY).name}/{DATASET(row.EXP_DATASET).name}/*.csv.xz",
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
            metric_path = Path(metric_path)

            metric_df = pd.read_csv(metric_path)
            metric_df = metric_df.loc[
                metric_df["EXP_UNIQUE_ID"] == row["EXP_UNIQUE_ID"]
            ]
            metric_df = metric_df.drop("EXP_UNIQUE_ID", axis=1).iloc[0]

            metric_df = metric_df.to_list()

            metric_name = str(metric_path.name).removesuffix(".csv.xz")

            metrics_data.append([metric_name, *metric_df])

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index(0, inplace=True)
        metrics_df = metrics_df.T
        print(metrics_df)
        pearson = metrics_df.corr(method="pearson")
        spearman = metrics_df.corr(method="spearman")
        corr = pearson

        # corr = corr[corr.columns[::-1]]

        # matrix = np.tril(corr)

        cm = sns.light_palette("green", as_cmap=True)
        sns.heatmap(
            corr, vmin=0, vmax=1, annot=True, cmap=cm, fmt=".1%"
        )  # , mask=matrix)

        plt.show()

        # TODO: return correlation values, average them!
        exit(-1)

    # dense_workload.parallel_apply(_do_stuff_parallel, axis=1)
    dense_workload.apply(_do_stuff_parallel, axis=1)


_calculate_correlations()
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
