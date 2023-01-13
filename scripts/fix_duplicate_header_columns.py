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
import shutil

pandarallel.initialize(progress_bar=True)
config = Config()
for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    metric_file = Path(file_name)
    tmp_metric_file = Path(str(metric_file) + ".tmp")

    if metric_file.name.endswith("_workload.csv"):
        continue

    with open(metric_file, "r") as mf:
        with open(tmp_metric_file, "w") as tmf:
            for ix, line in enumerate(mf):
                if ix == 0 or "EXP_UNIQUE_ID" not in line:
                    tmf.write(line)
                else:
                    print(metric_file)
    shutil.move(src=tmp_metric_file, dst=metric_file)
