from pathlib import Path
import sys
import glob

import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

config = Config()
for file_name in glob.glob(
    str(config.OUTPUT_PATH) + "/**/macro_f1-score.csv", recursive=True
):
    metric_file = Path(file_name)

    df = pd.read_csv(metric_file, header=0, delimiter=",")

    # delete every second column
    counter = 0
    if len(df.columns) < 22:
        print(metric_file)
        # exit(-1)
        continue
    for col_id in range(1, len(df.columns) - 1, 2):
        df = df.drop(str(col_id), axis=1)

        # rename columns
        df = df.rename(columns={str(col_id - 1): str(counter)})
        counter += 1

    df.to_csv(metric_file, index=False)
