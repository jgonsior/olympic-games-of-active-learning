# import modin.pandas as pd
import itertools
from turtle import done
import pandas as pd
from misc.config import Config
import ray
from tabulate import tabulate
from datasets import DATASET
from resources.data_types import (
    AL_STRATEGY,
    _convert_encrypted_strat_enum_to_readable_string,
)

# ray.init()
config = Config()

done_workload: pd.DataFrame = pd.read_csv(config.DONE_WORKLOAD_PATH)  # type: ignore
open_workload: pd.DataFrame = pd.read_csv(config.WORKLOAD_FILE_PATH)  # type: ignore


open_jobs = open_workload.loc[
    ~open_workload["EXP_UNIQUE_ID"].isin(done_workload["EXP_UNIQUE_ID"])
]
done_jobs = done_workload.loc[
    done_workload["EXP_UNIQUE_ID"].isin(open_workload["EXP_UNIQUE_ID"])
]


dataset_strat_counts = {}
datasets = open_workload["EXP_DATASET"].unique().tolist()
strategies = open_workload["EXP_STRATEGY"].unique().tolist()

for dataset, strat in itertools.product(datasets, strategies):
    dataset_strat_counts[(dataset, strat)] = 0

for dataset, strat in zip(done_jobs.EXP_DATASET, done_jobs.EXP_STRATEGY):
    dataset_strat_counts[(dataset, strat)] += 1

for dataset, strat in itertools.product(datasets, strategies):
    open_count = int(
        open_workload.loc[
            (open_workload["EXP_STRATEGY"] == strat)
            & (open_workload["EXP_DATASET"] == dataset)
        ].count()[0]
    )

    dataset_strat_counts[
        (dataset, strat)
    ] = f"{dataset_strat_counts[(dataset, strat)]}/{open_count}"

table_data = [
    [""]
    + [
        _convert_encrypted_strat_enum_to_readable_string(strat, config)
        for strat in strategies
    ]
] + [
    ([str(DATASET(dataset))[8:]] + [0 for strat in strategies]) for dataset in datasets
]

for (dataset, strat), count in dataset_strat_counts.items():
    # convert dataset and strat to indices
    dataset = datasets.index(dataset) + 1  # because of column headers
    strat = strategies.index(strat) + 1
    table_data[dataset][strat] = count

# sort columns, rows -> using pandas
table_data_df = pd.DataFrame(table_data)
table_data_df.columns = table_data_df.iloc[0]
table_data_df.drop(0, inplace=True)
table_data_df.set_index("", inplace=True)
table_data_df.sort_index(axis=0, inplace=True)
table_data_df.sort_index(axis=1, inplace=True)
print(table_data_df)
html = tabulate(table_data_df, headers="keys", tablefmt="html")
with open(config.HTML_STATUS_PATH, "w") as f:
    f.write(html)
