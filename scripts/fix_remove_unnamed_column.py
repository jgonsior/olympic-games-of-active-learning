import multiprocessing
from pathlib import Path
import sys
import glob
import lzma
from joblib import Parallel, delayed

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import shutil
import pandas as pd

config = Config()


glob_list = [
    f
    for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
    if not f.endswith("_workload.csv") and not f.endswith("_workloads.csv")
]
print(len(glob_list))


def _do_stuff(file_name):
    metric_file = Path(file_name)
    tmp_metric_file = Path(str(metric_file) + ".tmp")
    #  if not str(metric_file).endswith("ALIPY_UNCERTAINTY_MM/Iris/y_pred_test.csv.xz"):
    #  return

    with lzma.open(metric_file, "rt") as f:
        first_line = f.readline()

    if not "Unnamed: 0" in first_line:
        return

    print(metric_file)
    print("removing unnamed")

    df = pd.read_csv(metric_file)
    del df["Unnamed: 0"]
    df.to_csv(metric_file, index=False)


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    # Parallel(n_jobs=1, verbose=10)(
    delayed(_do_stuff)(file_name)
    for file_name in glob_list
)
