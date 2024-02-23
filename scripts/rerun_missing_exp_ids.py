import os
import sys
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
import multiprocessing

sys.dont_write_bytecode = True

from misc.config import Config


config = Config()

broken_exp_df = pd.read_csv(config.MISSING_EXP_IDS_IN_METRIC_FILES)


def run_code(i):
    cli = f"python 02_run_experiment.py --EXP_TITLE {config.EXP_TITLE} --WORKER_INDEX {i} --WORKLOAD_FILE_PATH {config.MISSING_EXP_IDS_IN_METRIC_FILES}"
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


# with parallel_backend("loky", n_jobs=8):
with parallel_backend("threading", n_jobs=multiprocessing.cpu_count()):
    Parallel()(delayed(run_code)(i) for i in range(0, len(broken_exp_df)))
