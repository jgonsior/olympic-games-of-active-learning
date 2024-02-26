import csv
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd
from sklearn import metrics
from tomlkit import TOMLDocument

from misc.helpers import append_and_create, get_df, get_glob_list

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
config = Config()

# iterate over all metric files
# check if a) metric files are all readable
# check b) if exp_unique_ids in metric files are all present in done/dense/failed/open/oom workloads


done_exp_ids = set(
    pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)["EXP_UNIQUE_ID"].to_list()
)
oom_exp_ids = set(
    pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)["EXP_UNIQUE_ID"].to_list()
)
failed_exp_ids = set(
    pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)["EXP_UNIQUE_ID"].to_list()
)
dense_exp_ids = set(pd.read_csv(config.DENSE_WORKLOAD_PATH)["EXP_UNIQUE_ID"].to_list())
open_exp_ids = set(pd.read_csv(config.WORKLOAD_FILE_PATH)["EXP_UNIQUE_ID"].to_list())

glob_list = get_glob_list(config)


def _do_stuff(file_name: Path):
    metric_df = get_df(file_name, config)

    if metric_df is None:
        return

    exp_ids_in_metric_file = set(metric_df["EXP_UNIQUE_ID"].to_list())

    if len(exp_ids_in_metric_file.difference(done_exp_ids)) > 0:
        append_and_create(
            config.MISSING_EXP_IDS_IN_METRIC_FILES,
            {
                "metric_file": file_name,
                "exp_ids": exp_ids_in_metric_file.difference(done_exp_ids),
                "intersection": "done",
            },
        )
    if len(exp_ids_in_metric_file.difference(oom_exp_ids)) > 0:
        append_and_create(
            config.MISSING_EXP_IDS_IN_METRIC_FILES,
            {
                "metric_file": file_name,
                "exp_ids": exp_ids_in_metric_file.difference(oom_exp_ids),
                "intersection": "oom",
            },
        )
    if len(exp_ids_in_metric_file.difference(failed_exp_ids)) > 0:
        append_and_create(
            config.MISSING_EXP_IDS_IN_METRIC_FILES,
            {
                "metric_file": file_name,
                "exp_ids": exp_ids_in_metric_file.difference(failed_exp_ids),
                "intersection": "failed",
            },
        )
    if len(exp_ids_in_metric_file.difference(dense_exp_ids)) > 0:
        append_and_create(
            config.MISSING_EXP_IDS_IN_METRIC_FILES,
            {
                "metric_file": file_name,
                "exp_ids": exp_ids_in_metric_file.difference(dense_exp_ids),
                "intersection": "dense",
            },
        )
    if len(exp_ids_in_metric_file.difference(open_exp_ids)) > 0:
        append_and_create(
            config.MISSING_EXP_IDS_IN_METRIC_FILES,
            {
                "metric_file": file_name,
                "exp_ids": exp_ids_in_metric_file.difference(open_exp_ids),
                "intersection": "open",
            },
        )


# Parallel(n_jobs=1, verbose=10)(
Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
