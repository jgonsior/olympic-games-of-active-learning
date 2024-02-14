import ast
import glob
import multiprocessing
from pathlib import Path
import sys

import dask.dataframe as dd
from joblib import Parallel, delayed
import pandas as pd
import functools

sys.dont_write_bytecode = True

from misc.config import Config


config = Config()


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return

        return inner

    return decorator


@with_timeout(100)
def _do_stuff(file_name):
    if Path(file_name + ".parquet").exists():
        return
    print(file_name)

    # from pandarallel import pandarallel

    # pandarallel.initialize(
    #    progress_bar=True, nb_workers=int(multiprocessing.cpu_count())
    # )

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

# Parallel(n_jobs=8, verbose=10)(
Parallel(n_jobs=int(multiprocessing.cpu_count()), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
