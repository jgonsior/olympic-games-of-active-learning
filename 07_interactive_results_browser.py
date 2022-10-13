import enum
from pathlib import Path
from typing import List
from flask import Flask, render_template
from datasets import DATASET
from interactive_results_browser.csv_helper_functions import (
    get_exp_config_names,
    get_exp_grid_request_params,
)
from interactive_results_browser.visualizations import (
    vizualization_to_python_function_mapping,
)

from livereload import Server
from interactive_results_browser.visualizations.base import Base_Visualizer
from misc.config import Config
from collections.abc import Iterable

from livereload.watcher import INotifyWatcher

app = Flask(
    __name__,
    template_folder="interactive_results_browser/templates",
    static_folder="interactive_results_browser/static",
)
app.debug = True


@app.route("/")
def show_available_experiments():

    config = Config(no_cli_args={"WORKER_INDEX": None})
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
    for viz in exp_grid_request_params["VISUALIZATIONS"]:
        visualizer = vizualization_to_python_function_mapping[viz](config)
        visualizations_and_tables.append(visualizer)

    return render_template(
        "viz.html.j2",
        experiment_name=experiment_name,
        config=config,
        full_exp_grid=full_exp_grid,
        visualizations_and_tables=visualizations_and_tables,
    )


if __name__ == "__main__":
    # check if static external ressources exist
    # if not: download them
    static_ressources = {
        "https://raw.githubusercontent.com/kevquirk/simple.css/main/simple.min.css": "_simple.min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/css/tom-select.css": "_tom_min.css",
        "https://cdn.jsdelivr.net/npm/tom-select@2.2.1/dist/js/tom-select.complete.min.js": "_tom_min.js",
    }

    for sr_url, sr_local_path in static_ressources.items():
        sr_local_path = Path(f"interactive_results_browser/static/{sr_local_path}")
        if not sr_local_path.exists():
            sr_local_path.write_bytes(requests.get(sr_url).content)

    server = Server(app.wsgi_app, watcher=INotifyWatcher())
    # server.watch("*.py")
    # server.watch("*.yaml")
    # server.watch("*.j2")
    # server.watch("*.css")
    # server.watch("*.js")
    server.serve()
