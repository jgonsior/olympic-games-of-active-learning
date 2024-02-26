import sys
import glob
import pandas as pd

from misc.helpers import get_df, get_glob_list

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing
from pathlib import Path

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


broken_csv_files = pd.read_csv(config.BROKEN_CSV_FILE_PATH)

for broken_csv_file in broken_csv_files["metric_file"]:
    broken_csv_file = Path(broken_csv_file)
    broken_csv_file.unlink()

    original_y_pred = Path(
        "/home/thiele/exp_results/bkp_01_full_exp_jan/"
        + str(broken_csv_file)
        .removesuffix(".parquet")
        .removeprefix("/home/thiele/exp_results/full_exp_jan/")
    )

    df = pd.read_csv(original_y_pred)
    df.to_parquet(broken_csv_file)
