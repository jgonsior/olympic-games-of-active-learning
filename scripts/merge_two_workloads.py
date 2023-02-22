import sys
import glob

import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
config = Config()

# ensure that EXP_UNIQUE_IDs are different between both workloads
# then merge the individual metric files together, but with new exp_unique_ids
# then concat done_workload_df and open_workload_df and failed_workload_df


exit(-1)
overtime_exp_unique_ids = []
for file_name in glob.glob(
    str(config.OUTPUT_PATH) + "/**/query_selection_time.csv", recursive=True
):
    df = pd.read_csv(file_name)
    col_names = [c for c in df.columns if c != "EXP_UNIQUE_ID"]

    for _, row in df.iterrows():
        if (row[col_names] > config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT).any():
            overtime_exp_unique_ids.append(row["EXP_UNIQUE_ID"])


for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(overtime_exp_unique_ids)]
    if len(df) < a:
        df.to_csv(file_name, index=False)
