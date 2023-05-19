import sys
import glob
import pandas as pd
from tqdm import tqdm

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from joblib import Parallel, delayed
import multiprocessing

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()

oom_workload = pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)
done_workload = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
failed_workload = pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)
print(failed_workload["EXP_UNIQUE_ID"])
print(oom_workload["EXP_UNIQUE_ID"])
# remove from oom_workload all present in done and in failed
oom_workload = oom_workload.loc[
    ~oom_workload["EXP_UNIQUE_ID"].isin(done_workload["EXP_UNIQUE_ID"])
]
oom_workload = oom_workload.loc[
    ~oom_workload["EXP_UNIQUE_ID"].isin(failed_workload["EXP_UNIQUE_ID"])
]

oom_workload_unique_ids = oom_workload["EXP_UNIQUE_ID"]
# 1349284
print(len(oom_workload_unique_ids))

glob_list = [
    f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
]


def _do_stuff(file_name):
    # print(file_name)
    try:
        df = pd.read_csv(file_name)
        a = len(df)
        df = df[~df["EXP_UNIQUE_ID"].isin(oom_workload_unique_ids)]
        if len(df) < a:
            print(f"{len(df)}<{a}: {file_name}")
            df.to_csv(file_name, index=False)
    except:
        print(f"ERROR: {file_name}")


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
