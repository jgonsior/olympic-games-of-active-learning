from collections import Counter
import glob
from os import dup
import sys

import pandas as pd
from sklearn.conftest import fetch_olivetti_faces


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


# create set of all prestent exp_ids
done_workload = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
failed_workload = pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)
started_oom_workloads = pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)


failed_workload.drop(columns="error", inplace=True)

exp_ids_present = set()

for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    # print(file_name)
    df = pd.read_csv(file_name, usecols=["EXP_UNIQUE_ID"])

    exp_ids_present = exp_ids_present.union(df["EXP_UNIQUE_ID"].to_list())

# print(exp_ids_present)


# check which params exist in these exp_ids
# check for duplicates there

combined_df = pd.concat([done_workload, failed_workload, started_oom_workloads])
combined_df = combined_df[combined_df["EXP_UNIQUE_ID"].isin(exp_ids_present)]

combined_df2 = combined_df.drop(columns="EXP_UNIQUE_ID")
duplicates = combined_df2.duplicated()

dup_exp_id_rename_counter = {}
for _, row in combined_df.iloc[duplicates[duplicates == True].index].iterrows():
    keys = list(row.keys())
    keys.remove("EXP_UNIQUE_ID")

    selector = True
    # find original exp_id
    for k, v in row.items():
        selector = selector & (combined_df[k] == v)

    exp_ids = combined_df.loc[selector]["EXP_UNIQUE_ID"].to_list()

    dup_exp_id_rename_counter[exp_ids[0]] = exp_ids[1:]

print(dup_exp_id_rename_counter)


exit(-1)


a = len(combined_df)

combined_df2 = combined_df.drop(columns="EXP_UNIQUE_ID")
print(combined_df2)

duplicates = combined_df2.duplicated()

dupl_counter = Counter(duplicates.to_list())
print(dupl_counter)

# if there are duplicates -> find out which exp_unique_ids these were
# TODO false durch TRUE ersetzen
if dupl_counter[False] > 0:
    combined_df_duplicates = combined_df.iloc[duplicates[duplicates == False].index]


# and then rename all lines in the metric files with them
# and then rename the exp_unique_ids in done_workload
# and then remove duplicates in the metric files

exit(-1)


duplicates = combined_df.duplicated()

duplicate_exp_unique_ids = combined_df[duplicates == True]["EXP_UNIQUE_ID"]

print(duplicate_exp_unique_ids)

for file_name in glob.glob(
    str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True
) + glob.glob(str(config.OUTPUT_PATH) + "*.csv", recursive=True):
    # print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(duplicate_exp_unique_ids)]
    if len(df[~df["EXP_UNIQUE_ID"].isin(duplicate_exp_unique_ids)]) < a:
        print(file_name)
        print(df[df["EXP_UNIQUE_ID"].isin(duplicate_exp_unique_ids)])
        # df.to_csv(file_name, index=False)
        print("would delete")


dupl_counter = Counter(duplicates.to_list())
print(dupl_counter)
