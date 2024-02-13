import csv
import multiprocessing
import sys
import glob
from joblib import Parallel, delayed

import pandas as pd
from datasets import DATASET



sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
config = Config()


datasets_to_remove = [
    DATASET[ddd]
    for ddd in [
        "wbc",
        "cylinder",
        "hepatitis",
        "arrythmia",
        "credit-approval",
        "sick",
        "eucalyptus",
        "jm1",
        "MiceProtein",
        "statlog_vehicle",
        "liver",
    ]
]

df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
exp_ids_to_remove = set(
    df[df["EXP_DATASET"].isin(datasets_to_remove)]["EXP_UNIQUE_ID"].tolist()
)

exp_ids_to_remove.add("5000244")
exp_ids_to_remove.add("4735538")
exp_ids_to_remove.add("3684905")

print(f"Removing {len(exp_ids_to_remove)}")


def _do_stuff(file_name):
    print(file_name)

    try:
        df = pd.read_csv(file_name)
        a = len(df)
        df = df[~df["EXP_UNIQUE_ID"].isin(exp_ids_to_remove)]
        if len(df) < a:
            df.to_csv(file_name, index=False, compression="infer")
    except Exception as err:
        exc_type, value, traceback = sys.exc_info()

        error = {
            "file_name": file_name,
            "exc_type": exc_type,
            "value": value,
            "traceback": traceback,
        }
        with open(config.WRONG_CSV_FILES_PATH, "a") as f:
            w = csv.DictWriter(f, fieldnames=error.keys())

            if config.WRONG_CSV_FILES_PATH.stat().st_size == 0:
                w.writeheader()
            w.writerow(error)


glob_list = [
    f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True)
]

glob_list = set(
    [
        *glob_list,
        *[f for f in glob.glob(str(config.OUTPUT_PATH) + "/*.csv", recursive=True)],
    ]
)
print(len(glob_list))


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    # Parallel(n_jobs=1, verbose=10)(
    delayed(_do_stuff)(file_name)
    for file_name in glob_list
)
