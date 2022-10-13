from __future__ import annotations
import enum
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

from typing import TYPE_CHECKING, Any, List
from matplotlib.pyplot import table
import numpy as np
from numpy import full
import pandas as pd
from sklearn.model_selection import ParameterGrid
from datasets import DATASET
from misc.helpers import _create_exp_grid
from resources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
    _convert_encrypted_strat_to_enum_param_tuple,
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
    get_data_exp_frame: Dict[str, List[str]],
) -> pd.DataFrame:
    if len(get_data_exp_frame) == 0:
        # do not filter at all!
        pass
    else:
        # only focus on the specified samples
        for k, v in get_data_exp_frame.items():
            k = k.replace("_GRID", "")

            if k == "EXP_STRATEGY":
                pass
            elif k == "EXP_LEARNER_MODEL":
                # convert into enum
                v = [int(LEARNER_MODEL[vv]) for vv in v]
            else:
                v = [int(vv) for vv in v]

            full_workload = full_workload.loc[full_workload[k].isin(v)]
            done_jobs = done_jobs.loc[done_jobs[k].isin(v)]

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

        dataset_strat_counts[(dataset, strat)] = (
            dataset_strat_counts[(dataset, strat)],
            open_count,
        )

    table_data = [
        [""]
        + [
            _convert_encrypted_strat_to_enum_param_tuple(strat, config)
            for strat in strategies
        ]
    ] + [([DATASET(dataset)] + [0 for strat in strategies]) for dataset in datasets]

    for (dataset, strat), count in dataset_strat_counts.items():
        # convert dataset and strat to indices
        dataset = datasets.index(dataset) + 1  # because of column headers
        strat = strategies.index(strat) + 1
        table_data[dataset][strat] = count

    # sort columns, rows -> using pandas
    table_data_df = pd.DataFrame(table_data)

    # sort after dataset NAME
    dataset_names = table_data_df[1:][0].values.tolist()
    dataset_names_str = [str(x.name).lower() for x in dataset_names]
    sorted_range = sorted(
        range(len(dataset_names_str)), key=dataset_names_str.__getitem__
    )
    dataset_names_ints = list(range(len(dataset_names_str)))
    new_sorting = [sorted_range.index(x) for x in dataset_names_ints]
    custom_sorting = {idx: value + 1 for idx, value in zip(dataset_names, new_sorting)}
    custom_sorting[""] = 0

    table_data_df.sort_values(
        by=[table_data_df.columns[0]],
        key=lambda x: x.map(lambda col: custom_sorting[col]),
        inplace=True,
    )

    # same for header columns
    column_names = table_data_df.columns.values.tolist()[1:]
    strategy_names = [str(x).lower() for x in table_data_df.loc[0][1:]]
    yx = list(zip(strategy_names, column_names))
    yx_sort = sorted(yx)
    sorted_column_titles = [0] + [x[1] for x in yx_sort]

    return table_data_df


def get_exp_grid(experiment_name: str, config: Config):
    exp_configs = yaml.safe_load(Path(config.LOCAL_YAML_EXP_PATH).read_bytes())
    exp_config = exp_configs[experiment_name]

    # strategy -> strategy enum and params

    exp_config["EXP_GRID_STRATEGY"] = _create_exp_grid(
        exp_config["EXP_GRID_STRATEGY"], config
    )

    def _al_strat_string_to_enum(encrtpyet_al_strat: str) -> str:
        splits = encrtpyet_al_strat.split(config._EXP_STRATEGY_STRAT_PARAMS_DELIM)
        splits[0] = str(int(AL_STRATEGY[splits[0]]))
        return config._EXP_STRATEGY_STRAT_PARAMS_DELIM.join(splits)

    exp_config["EXP_GRID_STRATEGY"] = [
        _al_strat_string_to_enum(k) for k in exp_config["EXP_GRID_STRATEGY"]
    ]

    exp_config["EXP_GRID_STRATEGY"] = [
        _convert_encrypted_strat_to_enum_param_tuple(param["x"], config)
        for param in ParameterGrid({"x": exp_config["EXP_GRID_STRATEGY"]})
    ]

    exp_config["EXP_GRID_DATASET"] = [
        DATASET(dataset_id) for dataset_id in exp_config["EXP_GRID_DATASET"]
    ]
    return exp_configs[experiment_name]
