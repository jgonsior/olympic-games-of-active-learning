import ast
import functools
import itertools
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config

from misc.helpers import get_df

config = Config()


# open .csv.xz and .csv
# concat
# deduplicate
# special handling with ast.literal_eval for .parquet files
# done?config = Config()


print(f"Merging {config.OUTPUT_PATH} and {config.SECOND_MERGE_PATH}")


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
def _do_stuff(exp_dataset, exp_strategy, config):
    csv_glob_list = sorted(
        [
            Path(ggg)
            for ggg in glob.glob(
                config.SECOND_MERGE_PATH
                + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv.xz",
                recursive=True,
            )
        ]
    )

    if len(csv_glob_list) == 0:
        return

    for csv_file_name in csv_glob_list:
        print(csv_file_name)
        csv_df = get_df(csv_file_name, config)

        if not "y_pred" in csv_file_name.name:
            continue

        if "y_pred" in csv_file_name.name:
            cols_with_indice_lists = csv_df.columns.difference(["EXP_UNIQUE_ID"])

            # from pandarallel import pandarallel

            # pandarallel.initialize(
            #    progress_bar=True, nb_workers=int(multiprocessing.cpu_count())
            # )
            csv_df[cols_with_indice_lists] = (
                csv_df[cols_with_indice_lists].fillna("[]")
                # .parallel_map(lambda x: ast.literal_eval(x))
                .map(lambda x: ast.literal_eval(x))
            )

        original_csv_path = (
            config.OUTPUT_PATH
            / csv_file_name.parent.parent.name
            / csv_file_name.parent.name
            / csv_file_name.name
        )

        if "y_pred" in csv_file_name.name:
            xz_df = get_df(Path(str(original_csv_path) + ".parquet"), config)
        else:
            xz_df = get_df(Path(str(original_csv_path)), config)

        xz_df = pd.concat([xz_df, csv_df], ignore_index=True).drop_duplicates(
            subset="EXP_UNIQUE_ID"
        )
        #  xz_df = csv_df

        #  if xz_df is
        if "Unnamed: " in xz_df.columns:
            del xz_df["Unnamed: 0"]

        if "y_pred" in csv_file_name.name:
            xz_df.to_parquet(Path(str(original_csv_path) + ".parquet"))
        else:
            xz_df.to_csv(original_csv_path, index=False)

        csv_file_name.unlink()


#  Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=int(multiprocessing.cpu_count()), verbose=10)(
    delayed(_do_stuff)(exp_dataset, exp_strategy, config)
    for (exp_dataset, exp_strategy) in itertools.product(
        config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY
    )
)


done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
second_done_workload_df = pd.read_csv(
    config.SECOND_MERGE_PATH + "/" + config.OVERALL_DONE_WORKLOAD_PATH.name
)

merged_workload_df = pd.concat(
    [done_workload_df, second_done_workload_df], ignore_index=True
).drop_duplicates(subset="EXP_UNIQUE_ID")

merged_workload_df.to_csv(config.OVERALL_DONE_WORKLOAD_PATH, index=False)
