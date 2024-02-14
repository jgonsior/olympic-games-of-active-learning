import ast
import glob
import multiprocessing
from pathlib import Path
import sys

import dask.dataframe as dd
from joblib import Parallel, delayed
import pandas as pd

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb)

sys.dont_write_bytecode = True

from misc.config import Config


config = Config()


def _do_stuff(file_name):
    a = pd.read_csv(file_name)
    cols_with_indice_lists = a.columns.difference(["EXP_UNIQUE_ID"])

    a[cols_with_indice_lists] = (
        a[cols_with_indice_lists].fillna("[]").map(lambda x: ast.literal_eval(x))
    )

    a.to_parquet(file_name + ".parquet")


glob_list = [
    *[
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH) + "/**/y_pred_train.csv.xz", recursive=True
        )
    ],
    *[
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH) + "/**/y_pred_test.csv.xz", recursive=True
        )
    ],
]

print(len(glob_list))

Parallel(n_jobs=1, verbose=10)(
    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name)
    for file_name in glob_list
)
