import sys

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import argparse
import multiprocessing
import os
import sys
from joblib import Parallel, delayed, parallel_backend

config = Config()

print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))


for computed_metric in config.COMPUTED_METRICS:
    print("#" * 100)
    print("computed_metric: " + str(computed_metric))
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()


"""def run_code(i):
    cli = f"python 02_run_experiment.py --EXP_TITLE local_SynDs --WORKER_INDEX {i}"
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
    Parallel()(delayed(run_code)(i) for i in range(0, 1680))"""
