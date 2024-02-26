import copy
import csv
import itertools
from pathlib import Path
import sys
import glob
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing

from datasets import DATASET
from misc.helpers import append_and_create, get_df, get_glob_list
from resources.data_types import AL_STRATEGY


sys.dont_write_bytecode = True

from misc.config import Config


config = Config()

dense_workload_df = pd.read_csv(config.DENSE_WORKLOAD_PATH)


datasets = [DATASET(int(ddd)) for ddd in dense_workload_df["EXP_DATASET"].unique()]
strats = [AL_STRATEGY(int(ddd)) for ddd in dense_workload_df["EXP_STRATEGY"].unique()]

print(len(datasets))
print(len(strats))

for ds in sorted(datasets, key=lambda v: v.name):
    print(f"{ds.name},")

for ds in sorted(strats, key=lambda k: k.name):
    print(f"{ds.name},")
