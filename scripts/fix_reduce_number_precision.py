import ast
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


glob_list = [
    f
    for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
    if not f.endswith("_workload.csv.xz") and not f.endswith("_workloads.csv.xz")
]
print(len(glob_list))


def _do_stuff(file_name):
    rounding_precision = 4

    metric_file = Path(file_name)
    tmp_metric_file = Path(str(metric_file) + ".tmp")
    #  if not str(metric_file).endswith("ALIPY_UNCERTAINTY_MM/Iris/y_pred_test.csv.xz"):
    #  return

    df = pd.read_csv(metric_file)

    if len(df) > 0 and "[" in str(df.iloc[0]):
        # print(metric_file)
        # print(df.head())
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")

        df = df.fillna("[]")

        try:
            df[column_names_which_are_al_cycles] = df[
                column_names_which_are_al_cycles
            ].map(
                lambda x: [
                    round(xxx, rounding_precision) for xxx in ast.literal_eval(str(x))
                ],
            )
        except TypeError:
            df = df.round(rounding_precision)
    else:
        df = df.round(rounding_precision)
        # print(df.head())
        # print("#" * 100)
        # print("\n" * 3)

    df.to_csv(file_name, index=False, compression="infer")


# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
