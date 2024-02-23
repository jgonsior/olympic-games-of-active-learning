import copy
import itertools
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

from misc.helpers import _append_and_create, _get_df, _get_glob_list

config = Config()


# open .csv.xz and .csv
# concat
# deduplicate
# special handling with ast.literal_eval for .parquet files
# done?config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)


def _do_stuff(exp_dataset, exp_strategy, config):
    csv_glob_list = sorted(
        [
            Path(ggg)
            for ggg in glob.glob(
                str(config.OUTPUT_PATH)
                + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv",
                recursive=True,
            )
        ]
    )

    if len(csv_glob_list) == 0:
        return

    for csv_file_name in csv_glob_list:
        csv_df = _get_df(csv_file_name, config)

        if "y_pred" in csv_file_name.name:
            xz_df = _get_df(Path(str(csv_file_name) + ".xz.parquet"), config)
        else:
            xz_df = _get_df(Path(str(csv_file_name) + ".xz"), config)


Parallel(n_jobs=1, verbose=10)(
    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(exp_dataset, exp_strategy, config)
    for (exp_dataset, exp_strategy) in itertools.product(
        config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY
    )
)
