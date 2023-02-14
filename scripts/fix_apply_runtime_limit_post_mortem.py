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

runtimes = []
max_time = 0
for file_name in glob.glob(
    str(config.OUTPUT_PATH) + "/**/query_selection_time.csv.xz", recursive=True
):
    df = pd.read_csv(file_name)
    col_names = [c for c in df.columns if c != "EXP_UNIQUE_ID"]
    df = df[col_names].to_numpy().flat
    for i in df:
        if i > max_time:
            max_time = i
            print(file_name, "\t\t\t", i)
    # runtimes = [*runtimes, *df]
    """metric_file = Path(file_name)
    tmp_metric_file = Path(str(metric_file) + ".tmp")


    with open(metric_file, "r") as mf:
        with open(tmp_metric_file, "w") as tmf:
            for ix, line in enumerate(mf):
                if ix == 0 or "EXP_UNIQUE_ID" not in line:
                    tmf.write(line)
                else:
                    print(metric_file)
    shutil.move(src=tmp_metric_file, dst=metric_file)
    """

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

runtimes = np.asarray(runtimes)
print(np.max(runtimes))
runtimes = np.where(runtimes > 600)
print(runtimes)
print(len(runtimes))
sns.histplot(runtimes, bins=30)
plt.show()
