from collections import Counter
import glob
import sys

import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


# read in all workloads
# append them
# remove exp_unique_ids
# check for duplicates

done_workload = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
failed_workload = pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)
started_oom_workloads = pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)

failed_workload.drop(columns="error", inplace=True)

combined_df = pd.concat([done_workload, failed_workload, started_oom_workloads])


duplicates = combined_df.duplicated()

duplicate_exp_unique_ids = combined_df[duplicates == True]["EXP_UNIQUE_ID"]

for file_name in glob.glob(
    str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True
) + glob.glob(str(config.OUTPUT_PATH) + "*.csv", recursive=True):
    print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(duplicate_exp_unique_ids)]
    if len(df) < a:
        # df.to_csv(file_name, index=False)
        print("would delete")


dupl_counter = Counter(duplicates.to_list())
print(dupl_counter)
