import sys
import glob
from matplotlib import use
from numpy import histogram_bin_edges
import pandas as pd
from tqdm import tqdm

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing
import dask.dataframe as dd
import numpy as np

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


# oom_workload = dd.read_csv(str(config.OVERALL_STARTED_OOM_WORKLOAD_PATH))
done_workload = dd.read_csv(
    config.OVERALL_DONE_WORKLOAD_PATH,
    dtype={
        "EXP_BATCH_SIZE": "float64",
        "EXP_DATASET": "float64",
        "EXP_LEARNER_MODEL": "float64",
        "EXP_NUM_QUERIES": "float64",
        "EXP_RANDOM_SEED": "float64",
        "EXP_START_POINT": "float64",
        "EXP_STRATEGY": "float64",
        "EXP_TRAIN_TEST_BUCKET_SIZE": "float64",
        "EXP_UNIQUE_ID": "float64",
    },
).astype(np.int64)
# failed_workload = dd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)

# ich nehme mir eine kombination aus batch_size, learner_model, start_points, train_test_bucket, dataset
# ann sollten für diese kombination alle strategien ergebnisse haben

column_combinations = [
    "EXP_DATASET",
    "EXP_STRATEGY",
    "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    "EXP_NUM_QUERIES",
    "EXP_BATCH_SIZE",
    "EXP_LEARNER_MODEL",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
]

#print(
#    done_workload.groupby(by=column_combinations)["EXP_UNIQUE_ID"].apply(list).compute()
#)


exp_ids_present_per_dataset = done_workload.groupby(by=[c for c in column_combinations if c != "EXP_DATASET"])["EXP_UNIQUE_ID"].apply(list).compute().to_frame()
print(exp_ids_present_per_dataset)
#exp_ids_present_per_dataset.to_parquet("exp_ids_present_per_dataset.parquet", engine="pyarrow")
exp_ids_present_per_strategy = done_workload.groupby(by=[c for c in column_combinations if c != "EXP_STRATEGY"])["EXP_UNIQUE_ID"].apply(list).compute().to_frame()
print(exp_ids_present_per_strategy)
#exp_ids_present_per_strategy.to_parquet("exp_ids_present_per_strategy.parquet", engine="pyarrow")

exit(-1)


exp_ids_present_in_all_metric_files = set()
exp_ids_present_per_dataset = {}
exp_ids_present_per_strategy = {}


glob_list = [
    f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
]


def _do_stuff(file_name):
    # print(file_name)
    try:
        df = pd.read_csv(file_name, usecols=["EXP_UNIQUE_ID"])
        exp_ids = set(df["EXP_UNIQUE_ID"].to_list())
        print(exp_ids)
    except Exception as e:
        print(f"ERROR: {file_name}")
        print(e)
        print("\n" * 2)


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
