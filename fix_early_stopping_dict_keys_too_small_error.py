from pathlib import Path
import sys
import os
import glob
import numpy as np

import pandas as pd

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

config = Config()
for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    metric_file = Path(file_name)

    if metric_file.name.endswith("_workload.csv"):
        continue

    try:
        df = pd.read_csv(metric_file, header=0, delimiter=",")
    except pd.errors.ParserError:
        print(metric_file)
        header = [iii for iii in range(0, config.EXP_GRID_NUM_QUERIES)]
        print(header)
        header.append("EXP_UNIQUE_ID")
        df = pd.read_csv(
            metric_file,
            header=0,
            names=header,
            on_bad_lines=lambda bl: [
                *[iii for iii in bl],
                *[np.nan for _ in range(len(bl), config.EXP_GRID_NUM_QUERIES)],
            ],
        )
    # print(df)
