import ast
import glob
import multiprocessing
from pathlib import Path
import sys

from joblib import Parallel, delayed
import pandas as pd
import functools

sys.dont_write_bytecode = True

from misc.config import Config


config = Config()


def _do_stuff(file_name):
    print(file_name)

    # from pandarallel import pandarallel

    # pandarallel.initialize(
    #    progress_bar=True, nb_workers=int(multiprocessing.cpu_count())
    # )

    a = pd.read_parquet(file_name)

    if type(a.iloc[0][0]) == str:
        print("oh oh")
    else:
        print(type(a.iloc[0][0]))
        return

    cols_with_indice_lists = a.columns.difference(["EXP_UNIQUE_ID"])

    a[cols_with_indice_lists] = (
        a[cols_with_indice_lists].fillna("[]").map(lambda x: ast.literal_eval(x))
    )
    # a.to_parquet(file_name)


glob_list = [
    *[
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH) + "/**/y_pred_*.csv.xz.parquet", recursive=True
        )
    ],
]

print(len(glob_list))

# Parallel(n_jobs=8, verbose=10)(
Parallel(n_jobs=int(multiprocessing.cpu_count()), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
