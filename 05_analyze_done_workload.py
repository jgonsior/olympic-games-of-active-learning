import ast
from itertools import chain
import multiprocessing
import timeit
import pandas as pd
import csv
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import pandas as pd
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed
from ressources.data_types import AL_STRATEGY
from joblib import Parallel, delayed

# easy example: extract a single metric from all random_seed_runs of all datasets and strategies to compare
# use t04_done_workload.csv as reference

config = Config()


#


METRIC_OF_INTEREST = "acc_auc"

# done_workload = pd.read_csv(    str(config.EXP_RESULT_EXTRACTED_ZIP_PATH / config.DONE_WORKLOAD_PATH.name))
done_workload = pd.read_csv(config.DONE_WORKLOAD_FILE)


done_workload["EXP_FULL_STRATEGY"] = (
    done_workload["EXP_STRATEGY"] + "#" + done_workload["EXP_STRATEGY_PARAMS"]
)

datasets = done_workload["EXP_DATASET"].unique()
al_strategies = done_workload["EXP_FULL_STRATEGY"].unique()
batch_sizes = done_workload["EXP_BATCH_SIZE"].unique()
learner_models = done_workload["EXP_LEARNER_MODEL"].unique()
train_test_buckets = done_workload["EXP_TRAIN_TEST_BUCKET_SIZE"].unique()

print(done_workload)

metrics = [
    "duration",
    "acc_auc",
    "macro_f1_auc",
    "macro_prec_auc",
    "macro_recall_auc",
    "weighted_f1_auc",
    "weighted_prec_auc",
    "weighted_recall_auc",
]

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
                    ].index

                    if len(ids_of_interest) == 0:
                        continue

                    rows = done_workload.iloc[ids_of_interest]
                    # print("\n" * 2)
                    # print(rows.iloc[0].to_dict())
                    # for metric in metrics:
                    #    print(f"{metric}: {rows[metric].mean()}")

# x aggregration as config
# x for all metrics
# table with the amount of repeated runs
# if wanted -> only keep results where we have all repetitions from all strategies
# table with signifcance tests
