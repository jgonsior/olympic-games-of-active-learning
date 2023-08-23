from pathlib import Path
import sys


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import pandas as pd

pandarallel.initialize(progress_bar=True)
import sys
from datasets import DATASET
from resources.data_types import AL_STRATEGY

config = Config()


df: pd.DataFrame = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
df["EXP_DATASET"] = df["EXP_DATASET"].apply(lambda x: DATASET(int(x)).name)
df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(lambda x: AL_STRATEGY(int(x)).name)

groupings = [
    ["EXP_DATASET", "EXP_STRATEGY"],
    ["EXP_DATASET"],
    ["EXP_STRATEGY"],
]


for grouping in groupings:
    df2 = (
        df.groupby(grouping)
        .size()
        .reset_index()
        .rename(columns={0: "count"})
        .sort_values("count")
    )
    print("\n" * 3)
    print(f"Group by {grouping} and sorted after count")
    print(df2)


result_df = None

# compute how long the already run strategies needed
for EXP_DATASET in df["EXP_DATASET"].unique():
    for EXP_STRATEGY in df["EXP_STRATEGY"].unique():
        METRIC_RESULTS_FILE = Path(
            config.OUTPUT_PATH
            / EXP_STRATEGY
            / EXP_DATASET
            / str("query_selection_time.csv.xz")
        )
        if not METRIC_RESULTS_FILE.exists():
            continue

        metric_df = pd.read_csv(METRIC_RESULTS_FILE, header=0, delimiter=",")
        metric_df["total_time"] = metric_df[
            [str(i) for i in range(0, len(metric_df.columns) - 1)]
        ].sum(axis=1)
        metric_df = metric_df[["total_time", "EXP_UNIQUE_ID"]]

        inner_product_df = df.merge(
            metric_df, how="inner", on="EXP_UNIQUE_ID", suffixes=(None, "r")
        )
        if result_df is None:
            result_df = inner_product_df
        else:
            result_df = pd.concat([result_df, inner_product_df])


groupings = [
    ["EXP_DATASET", "EXP_STRATEGY"],
    ["EXP_DATASET"],
    ["EXP_STRATEGY"],
]

groupings = [
    ["EXP_LEARNER_MODEL"],
    ["EXP_START_POINT"],
    ["EXP_BATCH_SIZE"],
    ["EXP_TRAIN_TEST_BUCKET_SIZE"],
]


for grouping in groupings:
    df2 = (
        result_df.groupby(grouping)
        .sum("total_time")
        .reset_index()
        .sort_values("total_time")
    )
    df2 = df2[[*grouping, *["total_time"]]]
    print(df2)
    df2.to_csv(f"test/{groupings}.csv")
