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

broken_files_csv_path = Path(config.OUTPUT_PATH / "07_broken_csvs.csv")


if not broken_files_csv_path.exists():
    with open(broken_files_csv_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=["metric_file"])
        w.writeheader()

glob_list = [
    f
    for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
    if not f.endswith("_workload.csv.xz") and not f.endswith("_workloads.csv.xz")
]

parsed_metric_csv_file = pd.read_csv(broken_files_csv_path)
parsed_metric_csvs = set(parsed_metric_csv_file["metric_file"].to_list())

print(len(glob_list))
glob_list = [ggg for ggg in glob_list if ggg not in parsed_metric_csvs]
print(len(glob_list))


def _do_stuff(file_name):
    metric_file = Path(file_name)
    try:
        df = pd.read_csv(metric_file)
        lll = len(df.columns)
    except EOFError as e:
        with open(broken_files_csv_path, "a") as f:
            w = csv.DictWriter(f, fieldnames=["metric_file"])
            w.writerow({"metric_file": metric_file})

        print(metric_file)


# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
