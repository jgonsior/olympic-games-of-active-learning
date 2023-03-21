import sys

import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

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

oom_workload.to_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH, index=False)
