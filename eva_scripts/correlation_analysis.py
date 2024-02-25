from collections import defaultdict
import itertools
from pathlib import Path
import sys
import glob
import warnings
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from datasets import DATASET

from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config
import numpy as np
import seaborn as sns

from scipy.stats import spearmanr


config = Config()


done_workload = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

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


def _calculate_correlations(param_to_evaluate):
    dense_workload = pd.read_csv(
        config.DENSE_WORKLOAD_PATH,
    )

    print(f"Original: {len(dense_workload)}")
    print(param_to_evaluate)

    dense_workload_grouped = dense_workload.groupby(
        by=[ddd for ddd in column_combinations if ddd != param_to_evaluate]
    ).apply(lambda r: list(zip(r[param_to_evaluate], r["EXP_UNIQUE_ID"])))

    print(
        f"Calculating correlations for {param_to_evaluate}: {len(dense_workload_grouped)}"
    )
    print(dense_workload_grouped)

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
                        (
                            np.nanmean(ast.literal_eval(str(vvv)))
                            if str(vvv) != "nan"
                            else np.nan
                        )
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


for cc in [
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_STRATEGY",
    # "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    # "EXP_NUM_QUERIES",
    "EXP_LEARNER_MODEL",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    # "METRICS",
]:
    _calculate_correlations(cc)
