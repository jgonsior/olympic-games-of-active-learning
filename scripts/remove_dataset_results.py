import sys
import glob

import pandas as pd
from datasets import DATASET

from resources.data_types import LEARNER_MODEL


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
config = Config()


datasets_to_remove = [
    DATASET[ddd]
    for ddd in [
        "wbc",
        "cylinder",
        "hepatitis",
        "arrythmia",
        "credit-approval",
        "sick",
        "eucalyptus",
        "jm1",
        "MiceProtein",
        "statlog_vehicle",
        "liver",
    ]
]

exp_ids_to_remove = []


df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
exp_ids_to_remove = df[df["EXP_DATASET"].isin(datasets_to_remove)]["EXP_UNIQUE_ID"]

for file_name in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv", recursive=True):
    print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(exp_ids_to_remove)]
    if len(df) < a:
        df.to_csv(file_name, index=False, compression="infer")


for file_name in glob.glob(str(config.OUTPUT_PATH) + "/*.csv", recursive=True):
    print(file_name)
    df = pd.read_csv(file_name)
    a = len(df)
    df = df[~df["EXP_UNIQUE_ID"].isin(exp_ids_to_remove)]
    if len(df) < a:
        df.to_csv(file_name, index=False, compression="infer")
