from pathlib import Path
import sys
from joblib import Parallel, delayed
import pandas as pd
import multiprocessing

from misc.helpers import get_glob_list


sys.dont_write_bytecode = True

from misc.config import Config


config = Config()


def _do_stuff(file_name, config):
    print(file_name)
    other_parquet_file = Path(
        str(file_name).removesuffix(".csv.xz.parquet") + ".csv.parquet"
    )
    if not other_parquet_file.exists():
        return
    dfa = pd.read_parquet(file_name)
    dfb = pd.read_parquet(other_parquet_file)

    df_merged = pd.concat([dfa, dfb], ignore_index=True).drop_duplicates(
        subset="EXP_UNIQUE_ID"
    )

    df_merged.to_parquet(file_name)
    other_parquet_file.unlink()


glob_list = get_glob_list(config, limit=f"*/*/y_pred_*")


#  Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name, config) for file_name in glob_list
)
