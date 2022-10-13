import ast
from itertools import chain
import multiprocessing
import timeit
import modin.pandas as pd
import csv
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import modin.pandas as pd
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed
from resources.data_types import AL_STRATEGY
from joblib import Parallel, delayed

# easy example: extract a single metric from all random_seed_runs of all datasets and strategies to compare
# use t04_done_workload.csv as reference

config = Config()

METRIC_OF_INTEREST = "acc_auc"

# zip = zipfile.ZipFile(str(config.OUTPUT_PATH) + ".zip")

done_workload = pd.read_csv(
    str(config.EXP_RESULT_EXTRACTED_ZIP_PATH / config.DONE_WORKLOAD_PATH.name)
)
print(len(done_workload))


datasets = done_workload["EXP_DATASET"].unique()

al_strategies = done_workload["EXP_FULL_STRATEGY"].unique()
batch_sizes = done_workload["EXP_BATCH_SIZE"].unique()
learner_models = done_workload["EXP_LEARNER_MODEL"].unique()
train_test_buckets = done_workload["EXP_TRAIN_TEST_BUCKET_SIZE"].unique()
print(done_workload)


def _extract_metrics(interesting_id: int):
    metric_df = pd.read_csv(
        f"{config.EXP_RESULT_EXTRACTED_ZIP_PATH}/{dataset}/{interesting_id}_metric_results.csv"
    )

    acc_auc = metric_df["accuracy"].sum() / len(metric_df)
    macro_f1_auc = metric_df["macro avg_f1-score"].sum() / len(metric_df)
    macro_prec_auc = metric_df["macro avg_precision"].sum() / len(metric_df)
    macro_recall_auc = metric_df["macro avg_recall"].sum() / len(metric_df)
    weighted_f1_auc = metric_df["weighted avg_f1-score"].sum() / len(metric_df)
    weighted_prec_auc = metric_df["weighted avg_precision"].sum() / len(metric_df)
    weighted_recall_auc = metric_df["weighted avg_recall"].sum() / len(metric_df)
    metric_df["selected_indices"] = metric_df["selected_indices"].apply(
        ast.literal_eval
    )
    selected_indices = list(
        chain.from_iterable(metric_df["selected_indices"].to_list())
    )

    return {
        "dataset": dataset,
        "al_strategy": al_strategy,
        "batch_size": batch_size,
        "learner_model": learner_model,
        "train_test_bucket": train_test_bucket,
        "acc_auc": acc_auc,
        "macro_f1_auc": macro_f1_auc,
        "macro_prec_auc": macro_prec_auc,
        "macro_recall_auc": macro_recall_auc,
        "weighted_f1_auc": weighted_f1_auc,
        "weighted_prec_auc": weighted_prec_auc,
        "weighted_recall_auc": weighted_recall_auc,
        "selected_indices": selected_indices,
    }


with Parallel(n_jobs=multiprocessing.cpu_count()) as parallel:

    counter = 0
    results = []

    for batch_size in batch_sizes:
        for learner_model in learner_models:
            for train_test_bucket in train_test_buckets:
                # print(f"{batch_size} - {learner_model} - {train_test_bucket}")
                for al_strategy in al_strategies:
                    for dataset in datasets:
                        ids_of_interest = done_workload.loc[
                            (done_workload["EXP_DATASET"] == dataset)
                            & (done_workload["EXP_FULL_STRATEGY"] == al_strategy)
                            & (done_workload["EXP_BATCH_SIZE"] == batch_size)
                            & (done_workload["EXP_LEARNER_MODEL"] == learner_model)
                            & (
                                done_workload["EXP_TRAIN_TEST_BUCKET_SIZE"]
                                == train_test_bucket
                            )
                        ]["EXP_UNIQUE_ID"].to_list()

                        dataset = dataset.replace("DATASET.", "")

                        if len(ids_of_interest) > 1:
                            counter += 1
                            print(f"{counter} - {len(ids_of_interest)}")
                            temp_results = parallel(
                                delayed(_extract_metrics)(interesting_id)
                                for interesting_id in ids_of_interest
                            )
                            results += temp_results
                            """
                            if counter == 10:
                                results_df = pd.DataFrame(
                                    results,
                                )
                                print(results_df)
                                results_df.to_csv("test.csv", index=None)
                                exit(-1)
                            """
                # create table and save/display it

results_df = pd.DataFrame(
    results,
)
results_df.to_csv("test.csv", index=None)

# -> weg von zip -> ist das so schneller?!
# -> außerdem -> multithreading nutzen!
# read data into dataframe

# aggregration as config
# for all metrics
# table with the amount of repeated runs
# if wanted -> only keep results where we have all repetitions from all strategies
# table with signifcance tests
