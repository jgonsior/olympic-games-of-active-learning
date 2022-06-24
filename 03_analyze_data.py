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
import tarfile
from ressources.data_types import AL_STRATEGY
import zipfile

# easy example: extract a single metric from all random_seed_runs of all datasets and strategies to compare
# use t04_done_workload.csv as reference

config = Config()

METRIC_OF_INTEREST = "acc_auc"

df = pd.DataFrame()

zip = zipfile.ZipFile(str(config.OUTPUT_PATH) + ".zip")

done_workload = pd.read_csv(
    zip.open(str(config.EXP_RESULT_ZIP_PATH_PREFIX / config.DONE_WORKLOAD_PATH.name))
)
print(len(done_workload))


done_workload["EXP_FULL_STRATEGY"] = (
    done_workload["EXP_STRATEGY"] + "#" + done_workload["EXP_STRATEGY_PARAMS"]
)

datasets = done_workload["EXP_DATASET"].unique()

al_strategies = done_workload["EXP_FULL_STRATEGY"].unique()
batch_sizes = done_workload["EXP_BATCH_SIZE"].unique()
learner_models = done_workload["EXP_LEARNER_MODEL"].unique()
train_test_buckets = done_workload["EXP_TRAIN_TEST_BUCKET_SIZE"].unique()
print(done_workload)

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
                        print(ids_of_interest)
                        for interesting_id in ids_of_interest:
                            f = zip.open(
                                f"{config.EXP_RESULT_ZIP_PATH_PREFIX}/{dataset}/{interesting_id}_metric_results.csv"
                            )
                            # print(pd.read_csv(f))

            # create table and save/display it


# read data into dataframe

# aggregration as config
# for all metrics
# table with the amount of repeated runs
# if wanted -> only keep results where we have all repetitions from all strategies
# table with signifcance tests
