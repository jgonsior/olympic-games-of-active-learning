from __future__ import annotations
from pathlib import Path
from typing import List

from typing import List, TYPE_CHECKING

import requests

from datasets import DATASET
from analyse_results.visualizations import VISUALIZATION
from resources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
)
from jinja2 import Environment, PackageLoader

import yaml

if TYPE_CHECKING:
    from misc.config import Config


def _download_static_ressources():
    # check if static external ressources exist
    # if not: download them
    static_resources = {
        "https://raw.githubusercontent.com/kevquirk/simple.css/main/simple.min.css": "_simple.min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/css/tom-select.css": "_tom_min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/js/tom-select.complete.min.js": "_tom_min.js",
    }

    for sr_url, sr_local in static_resources.items():
        sr_local_path: Path = Path(f"analyse_results/static/{sr_local}")
        if not sr_local_path.exists():
            sr_local_path.write_bytes(requests.get(sr_url).content)


def get_exp_config_names(config: Config) -> List[str]:
    yaml_config_params = yaml.safe_load(Path(config.LOCAL_YAML_EXP_PATH).read_bytes())
    return yaml_config_params.keys()


def url_for(type: str = "static", filename: str = "", **params):
    return f"analyse_results/static/{filename}"


def get_exp_grid_without_flask_params(config: Config):
    full_exp_grid = {
        "EXP_BATCH_SIZE": config.EXP_GRID_BATCH_SIZE,
        "EXP_DATASET": config.EXP_GRID_DATASET,
        "EXP_LEARNER_MODEL": config.EXP_GRID_LEARNER_MODEL,
        "EXP_NUM_QUERIES": config.EXP_GRID_NUM_QUERIES,
        "EXP_STRATEGY": config.EXP_GRID_STRATEGY,
        "EXP_TRAIN_TEST_BUCKET_SIZE": config.EXP_GRID_TRAIN_TEST_BUCKET_SIZE,
        "EXP_NUM_QUERIES": config.EXP_GRID_NUM_QUERIES,
        "EXP_RANDOM_SEED": config.EXP_GRID_RANDOM_SEED,
        "EXP_START_POINT": config.EXP_GRID_START_POINT,
    }

    full_exp_grid["VISUALIZATIONS"] = [viz for viz in list(VISUALIZATION)]

    get_exp_grid_request_params = full_exp_grid

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


def render_template(template_url: str, **data: dict) -> str:
    env = Environment(loader=PackageLoader("analyse_results"))
    return env.get_template(template_url).render(
        **data,
        url_for=url_for,
    )
