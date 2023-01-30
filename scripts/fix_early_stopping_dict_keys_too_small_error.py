from pathlib import Path
import sys
import glob
import numpy as np

import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

config = Config()
for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    print(file_name)
    metric_file = Path(file_name)

    if metric_file.name.endswith("_workload.csv"):
        continue

    def _fix_nan_rows(row):
        if row.isnull().values.any():
            no_null_row = row[~row.isnull()]
            row[len(no_null_row) - 1] = np.nan
            row[["EXP_UNIQUE_ID"]] = int(no_null_row.iloc[-1])
        return row

    try:
        df = pd.read_csv(metric_file, header=0, delimiter=",")
    except pd.errors.ParserError:
        print(metric_file)
        header = [iii for iii in range(0, config.EXP_GRID_NUM_QUERIES[0])]
        header.append("EXP_UNIQUE_ID")
        df = pd.read_csv(
            metric_file,
            header=None,
            skiprows=1,
            engine="python",
            names=header,
            on_bad_lines=lambda bl: [
                *[aaa for aaa in bl],
                *[np.nan for _ in range(len(bl), config.EXP_GRID_NUM_QUERIES[0])],
            ],
        )

        df = df.apply(_fix_nan_rows, axis=1)
        df["EXP_UNIQUE_ID"] = df["EXP_UNIQUE_ID"].astype(int)
        df.to_csv(metric_file, index=False)

    if df["EXP_UNIQUE_ID"].isnull().values.any():
        df = df.apply(_fix_nan_rows, axis=1)
        df["EXP_UNIQUE_ID"] = df["EXP_UNIQUE_ID"].astype(int)
        df.to_csv(metric_file, index=False)
