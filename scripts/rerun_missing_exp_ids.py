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
    print(
        "TODO: 02_run_experiment muss ich per CLI argument übergeben, dass nicht in 01_workload.csv sondern in 06_broken_exp_id.csv geschaut wird"
    )
    return
    cli = f"timeout {config.SLURM_TIME_LIMIT} python 02_run_experiment.py --EXP_TITLE {config.EXP_TITLE} --WORKER_INDEX {i}"
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=1):
    # with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
    Parallel()(delayed(run_code)(i) for i in range(0, len(broken_exp_df)))
