from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from typing import TYPE_CHECKING, Any, List
import pandas as pd
from flask import request

from datasets import DATASET
from interactive_results_browser.visualizations import VISUALIZATION
from resources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
)

import yaml

if TYPE_CHECKING:
    from misc.config import Config


def get_exp_config_names(config: Config) -> List[str]:
    yaml_config_params = yaml.safe_load(Path(config.LOCAL_YAML_EXP_PATH).read_bytes())
    return yaml_config_params.keys()


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

    full_exp_grid["VISUALIZATIONS"] = [viz for viz in list(VISUALIZATION)]

    get_exp_grid_request_params = {}

    keys = list(set([*full_exp_grid.keys(), *request.args.keys()]))
    for k in keys:
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
