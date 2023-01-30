import sys

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel
import pandas as pd

pandarallel.initialize(progress_bar=True)
import sys
from datasets import DATASET
from resources.data_types import AL_STRATEGY

config = Config()


df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

df = (
    df.groupby(["EXP_DATASET", "EXP_STRATEGY"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    .sort_values("count")
)
df["EXP_DATASET"] = df["EXP_DATASET"].apply(lambda x: DATASET(int(x)).name)
df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(lambda x: AL_STRATEGY(int(x)).name)
print(df)
df.to_csv("test.csv", index=False)
