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

app = Flask(
    __name__,
    template_folder="interactive_results_browser/templates",
    static_folder="interactive_results_browser/static",
)
app.debug = True


# cache.init_app(app)


@app.route("/")
# @cache.cached(timeout=50)
def show_available_experiments():
    config = Config(no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": "only_random"})
    config_names = get_exp_config_names(config)
    return render_template("available_experiments.html.j2", config_names=config_names)


@app.route("/viz/<string:experiment_name>", methods=["GET"])
def show_results(experiment_name: str):
    config = Config(
        no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name},
    )

    exp_grid_request_params, full_exp_grid = get_exp_grid_request_params(
        experiment_name, config
    )

    visualizations_and_tables: List[Base_Visualizer] = []

    plt.ioff()

    for viz in exp_grid_request_params["VISUALIZATIONS"]:
        visualizer = vizualization_to_python_function_mapping[viz](
            config, exp_grid_request_params, experiment_name
        )
        visualizations_and_tables.append(visualizer)

    for viz in visualizations_and_tables:
        additional_request_params = viz.get_additional_request_params()
        if additional_request_params != {}:
            full_exp_grid = {
                **full_exp_grid,
                **additional_request_params,
            }

    return render_template(
        "viz.html.j2",
        experiment_name=experiment_name,
        config=config,
        full_exp_grid=full_exp_grid,
        visualizations_and_tables=visualizations_and_tables,
        isinstance=isinstance,
        Iterable=Iterable,
        Enum=Enum,
        int=int,
        str=str,
        exp_grid_request_params=exp_grid_request_params,
    )


if __name__ == "__main__":
    pandarallel.initialize(progress_bar=True, use_memory_fs=True)
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

    server = Server(app.wsgi_app, watcher=INotifyWatcher())
    server.serve()
