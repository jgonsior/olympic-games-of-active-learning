from __future__ import annotations
import itertools
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from typing import TYPE_CHECKING, Any, List
from enum import IntEnum, unique
from matplotlib.pyplot import table
import numpy as np
from numpy import full
import pandas as pd
from flask import request

from sklearn.model_selection import ParameterGrid
from datasets import DATASET
from interactive_results_browser.visualizations.visualizations import (
    _plot_learning_curves,
    _plot_retrieved_samples,
    _plot_run_done_stats,
)
from resources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
)

import yaml

if TYPE_CHECKING:
    from misc.config import Config


@unique
class VISUALIZATION(IntEnum):
    LEARNING_CURVES = 1
    RETRIEVED_SAMPLES = 2
    RUN_DONE_STATS = 3
    STRATEGY_RANKING = 4


vizualization_to_python_function_mapping: Dict[VISUALIZATION, Callable] = {
    VISUALIZATION.LEARNING_CURVES: _plot_learning_curves,
    VISUALIZATION.RETRIEVED_SAMPLES: _plot_retrieved_samples,
    VISUALIZATION.RUN_DONE_STATS: _plot_run_done_stats,
}


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
    exp_grid_request_params: Dict[str, List[str]],
) -> pd.DataFrame:
    print(exp_grid_request_params.keys())
    if len(exp_grid_request_params) == 0:
        # do not filter at all!
        pass
    else:
        # only focus on the specified samples
        for k, v in exp_grid_request_params.items():
            k = k.replace("_GRID", "")

            if k in ["VISUALIZATIONS"]:
                continue

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

    table_data = [[""] + [strat for strat in strategies]] + [
        ([DATASET(dataset)] + [0 for strat in strategies]) for dataset in datasets
    ]

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


def get_exp_grid_request_params(experiment_name: str, config: Config):
    full_exp_grid = {
        "EXP_BATCH_SIZE": config.EXP_GRID_BATCH_SIZE,
        "EXP_DATASET": config.EXP_GRID_DATASET,
        "EXP_LEARNER_MODEL": config.EXP_GRID_LEARNER_MODEL,
        "EXP_NUM_QUERIES": config.EXP_GRID_NUM_QUERIES,
        "EXP_STRATEGY": config.EXP_GRID_STRATEGY,
        "EXP_TRAIN_TEST_BUCKET_SIZE": config.EXP_GRID_TRAIN_TEST_BUCKET_SIZE,
        "EXP_NUM_QUERIES": config.EXP_GRID_NUM_QUERIES,
        "EXP_RANDOM_SEED": config.EXP_GRID_RANDOM_SEED,
    }

    print("Batch size")
    print(full_exp_grid["EXP_BATCH_SIZE"])

    full_exp_grid["VISUALIZATIONS"] = [viz for viz in list(VISUALIZATION)]

    get_exp_grid_request_params = {}

    for k in full_exp_grid.keys():
        if k in request.args.keys():
            try:
                get_exp_grid_request_params[k] = [
                    int(kkk) for kkk in request.args.getlist(k)
                ]
            except ValueError:
                get_exp_grid_request_params[k] = request.args.getlist(k)
        else:
            get_exp_grid_request_params[k] = full_exp_grid[k]

    # convert int_enums to real enums
    get_exp_grid_request_params["EXP_DATASET"] = [
        DATASET(int(dataset_id))
        for dataset_id in get_exp_grid_request_params["EXP_DATASET"]
    ]

    get_exp_grid_request_params["EXP_LEARNER_MODEL"] = [
        LEARNER_MODEL(int(dataset_id))
        for dataset_id in get_exp_grid_request_params["EXP_LEARNER_MODEL"]
    ]

    get_exp_grid_request_params["EXP_STRATEGY"] = [
        AL_STRATEGY(int(dataset_id))
        for dataset_id in get_exp_grid_request_params["EXP_STRATEGY"]
    ]

    get_exp_grid_request_params["VISUALIZATIONS"] = [
        VISUALIZATION(int(dataset_id))
        for dataset_id in get_exp_grid_request_params["VISUALIZATIONS"]
    ]

    return get_exp_grid_request_params, full_exp_grid
