import sys
import glob
import pandas as pd

from misc.helpers import _get_df, _get_glob_list

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()

glob_list = _get_glob_list(config)

glob_list.append(config.OVERALL_DONE_WORKLOAD_PATH)
glob_list.append(config.OVERALL_FAILED_WORKLOAD_PATH)
glob_list.append(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)
glob_list.append(config.WORKLOAD_FILE_PATH)

dense_workload = pd.read_csv(config.DENSE_WORKLOAD_PATH)
dense_ids = set(dense_workload.EXP_UNIQUE_ID.to_list())
print(len(dense_ids))
print(len(glob_list))


def _do_stuff(file_name: Path):
    # print(file_name)
    df = _get_df(file_name, config)

    if df is None:
        return

    a = len(df)
    df = df[df["EXP_UNIQUE_ID"].isin(dense_ids)]

    df = df.loc[~df.duplicated(subset="EXP_UNIQUE_ID")]

    if len(df) < a:
        print(f"{len(df)}<{a}: {file_name}")

        file_name = str(file_name)

        if len(df) == 0:
            Path(file_name).unlink()
        elif file_name.endswith(".csv.xz") or file_name.endswith(".csv"):
            df.to_csv(file_name, index=False)
        elif file_name.endswith(".parquet"):
            df.to_parquet(file_name)
        else:
            print("EEEEEEEEHHHHHHHHHHHHHHHHh")
            print(file_name)


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
