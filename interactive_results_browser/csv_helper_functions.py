from __future__ import annotations
import itertools
from pathlib import Path
from typing import List, Tuple

from typing import TYPE_CHECKING, Any, List
import pandas as pd
from datasets import DATASET
from resources.data_types import (
    AL_STRATEGY,
    _convert_encrypted_strat_enum_to_readable_string,
)

import yaml

if TYPE_CHECKING:
    from misc.config import Config


def get_exp_config_names(config: Config) -> List[str]:
    yaml_config_params = yaml.safe_load(Path(config.LOCAL_YAML_EXP_PATH).read_bytes())
    return yaml_config_params.keys()


def load_workload_csv_files(
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    done_workload: pd.DataFrame = pd.read_csv(config.DONE_WORKLOAD_PATH)  # type: ignore
    open_workload: pd.DataFrame = pd.read_csv(config.WORKLOAD_FILE_PATH)  # type: ignore
    full_workload = pd.concat([done_workload, open_workload])
    full_workload.drop_duplicates(subset="EXP_UNIQUE_ID", inplace=True)

    open_jobs = full_workload.loc[
        full_workload["EXP_UNIQUE_ID"].isin(open_workload["EXP_UNIQUE_ID"])
    ]
    done_jobs = full_workload.loc[
        full_workload["EXP_UNIQUE_ID"].isin(done_workload["EXP_UNIQUE_ID"])
    ]

    return full_workload, open_jobs, done_jobs


def create_open_done_workload_table(
    full_workload: pd.DataFrame,
    open_jobs: pd.DataFrame,
    done_jobs: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    dataset_strat_counts = {}
    datasets = full_workload["EXP_DATASET"].unique().tolist()
    strategies = full_workload["EXP_STRATEGY"].unique().tolist()

    for dataset, strat in itertools.product(datasets, strategies):
        dataset_strat_counts[(dataset, strat)] = 0

    for dataset, strat in zip(done_jobs.EXP_DATASET, done_jobs.EXP_STRATEGY):
        dataset_strat_counts[(dataset, strat)] += 1

    for dataset, strat in itertools.product(datasets, strategies):
        open_count = int(
            full_workload.loc[
                (full_workload["EXP_STRATEGY"] == strat)
                & (full_workload["EXP_DATASET"] == dataset)
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
        ([str(DATASET(dataset))[8:]] + [0 for strat in strategies])
        for dataset in datasets
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
    return table_data_df
