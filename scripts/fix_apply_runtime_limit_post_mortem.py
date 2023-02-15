from pathlib import Path
import sys
import glob
from matplotlib import pyplot as plt

import pandas as pd


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import shutil

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()


# if one batch takes longer -> the whole EXP_UNIQUE_ID gets nullified -> across all metrics!

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
