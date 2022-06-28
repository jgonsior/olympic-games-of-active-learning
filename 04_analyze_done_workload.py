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

METRIC_OF_INTEREST = "acc_auc"

# zip = zipfile.ZipFile(str(config.OUTPUT_PATH) + ".zip")

done_workload = pd.read_csv(
    str(config.EXP_RESULT_EXTRACTED_ZIP_PATH / config.DONE_WORKLOAD_PATH.name)
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


# -> weg von zip -> ist das so schneller?!
# -> außerdem -> multithreading nutzen!
# read data into dataframe

# aggregration as config
# for all metrics
# table with the amount of repeated runs
# if wanted -> only keep results where we have all repetitions from all strategies
# table with signifcance tests
