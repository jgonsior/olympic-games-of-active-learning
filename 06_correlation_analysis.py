from pathlib import Path
from typing import List
from enum import Enum
import requests
from flask import Flask, render_template
from interactive_results_browser.csv_helper_functions import (
    get_exp_config_names,
    get_exp_grid_request_params,
)
from interactive_results_browser.visualizations import (
    vizualization_to_python_function_mapping,
)

from livereload import Server
from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from misc.config import Config
from collections.abc import Iterable

from livereload.watcher import INotifyWatcher
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from flask_frozen import Freezer

from interactive_results_browser import app

pandarallel.initialize(progress_bar=False, use_memory_fs=True)
# check if static external ressources exist
# if not: download them
static_resources = {
    "https://raw.githubusercontent.com/kevquirk/simple.css/main/simple.min.css": "_simple.min.css",
    "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/css/tom-select.css": "_tom_min.css",
    "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/js/tom-select.complete.min.js": "_tom_min.js",
}

for sr_url, sr_local_path in static_resources.items():
    sr_local_path = Path(f"interactive_results_browser/static/{sr_local_path}")
    if not sr_local_path.exists():
        sr_local_path.write_bytes(requests.get(sr_url).content)

freezer = Freezer(app, log_url_for=False)


@freezer.register_generator
def find_urls():
    yield "/"


freezer.freeze()
