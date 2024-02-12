import ast
import csv
import multiprocessing
from pathlib import Path
import sys
import glob
import lzma

from joblib import Parallel, delayed
import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import shutil

pandarallel.initialize(progress_bar=True)
config = Config()

parsed_metric_csv_file_path = Path(config.OUTPUT_PATH / "07_parsed_metric_csvs.csv")


glob_list = [
    f
    for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
    if not f.endswith("_workload.csv.xz") and not f.endswith("_workloads.csv.xz")
]


def _do_stuff(file_name):
    metric_file = Path(file_name)
    try:
        df = pd.read_csv(metric_file)
        lll = len(df.columns)
    except EOFError as e:
        print(metric_file)


# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
