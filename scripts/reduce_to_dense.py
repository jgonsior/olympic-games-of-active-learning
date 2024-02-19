import sys
import glob
import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


glob_list = [
    *[f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)],
    *[
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH) + "/**/*.csv.xz.parquet", recursive=True
        )
    ],
    *[f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True)],
]

dense_workload = pd.read_csv(config.DENSE_WORKLOAD_PATH)
dense_ids = set(dense_workload.EXP_UNIQUE_ID.to_list())
print(len(dense_ids))
print(len(glob_list))


def _do_stuff(file_name: str):
    try:
        if file_name.endswith(".csv.xz") or file_name.endswith(".csv"):
            df = pd.read_csv(file_name)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(file_name)
        a = len(df)
        df = df[~df["EXP_UNIQUE_ID"].isin(dense_ids)]
        if len(df) < a:
            print(f"{len(df)}<{a}: {file_name}")

            if len(df) == 0:
                Path(file_name).unlink()
            elif file_name.endswith(".csv.xz") or file_name.endswith(".csv"):
                df.to_csv(file_name, index=False)
            elif file_name.endswith(".parquet"):
                df.to_parquet(file_name)
    except:
        print(f"ERROR: {file_name}")


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
