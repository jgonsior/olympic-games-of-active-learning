import sys
import glob

import pandas as pd

from resources.data_types import LEARNER_MODEL


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

# all batches which have been running longer than 10 minutes will be ignored

pandarallel.initialize(progress_bar=True)
config = Config()

lbfgs_mlp_exp_ids = []


df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
lbfgs_mlp_exp_ids = df[df["EXP_LEARNER_MODEL"] == LEARNER_MODEL.LBFGS_MLP.value][
    "EXP_UNIQUE_ID"
]

for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(lbfgs_mlp_exp_ids)]
    if len(df) < a:
        df.to_csv(file_name, index=False)
