import enum
from multiprocessing.sharedctypes import Value
from pathlib import Path
from pkgutil import get_data
from flask import Flask, render_template
from flask import request
import requests
from sklearn.datasets import fetch_olivetti_faces
from datasets import DATASET
from interactive_results_browser.csv_helper_functions import (
    create_open_done_workload_table,
    get_exp_config_names,
    get_exp_grid,
    load_workload_csv_files,
)
from livereload import Server
from misc.config import Config
from collections.abc import Iterable

from resources.data_types import _encrypt_tuple, _format_tuple
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


@app.route("/workload/<string:experiment_name>", methods=["GET"])
def show_open_done_workload(experiment_name: str):
    # random seed fehlt
    config = Config(no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name})

    exp_grid = get_exp_grid(experiment_name, config)
    get_data_exp_grid = {}

    exp_grid["EXP_GRID_RANDOM_SEED"] = list(
        range(0, exp_grid["EXP_GRID_RANDOM_SEEDS_END"])
    )

    for k in exp_grid.keys():
        if k in request.args.keys():
            try:
                get_data_exp_grid[k] = [int(kkk) for kkk in request.args.getlist(k)]
            except ValueError:
                get_data_exp_grid[k] = request.args.getlist(k)

    if "EXP_GRID_DATASET" in get_data_exp_grid:
        get_data_exp_grid["EXP_GRID_DATASET"] = [
            DATASET(int(dataset_id))
            for dataset_id in get_data_exp_grid["EXP_GRID_DATASET"]
        ]

    full_workload, open_jobs, done_jobs = load_workload_csv_files(config)

    open_done_df = create_open_done_workload_table(
        full_workload,
        open_jobs,
        done_jobs,
        config,
        get_data_exp_grid,
    )

    rows = list(open_done_df.values.tolist())
    rows[0][0] = "Dataset"

    return render_template(
        "open_done_workload.html.j2",
        experiment_name=experiment_name,
        column_names=rows[0],
        row_data=rows[1:],
        link_column="Dataset",
        zip=zip,
        str=str,
        isinstance=isinstance,
        Iterable=Iterable,
        exp_grid=exp_grid,
        get_data_exp_grid=get_data_exp_grid,
        type=type,
        tuple=tuple,
        enum=enum.Enum,
        int=int,
        _encrypt_tuple=_encrypt_tuple,
        _format_tuple=_format_tuple,
        config=config,
    )


@app.route("/dataset/<int:dataset_id>", methods=["GET"])
def show_dataset_overview(dataset_id: int):

    # dataset_name = dataset_id_to_name(dataset_id)
    # random_ids = random_ids_for_dataset(dataset_id)
    return render_template("dataset_overview.html.j2")


@app.route("/strategy/<int:strategy_id>", methods=["GET"])
def show_strategy_overview(strategy_id: int):
    return render_template("strategy_overview.html.j2")


@app.route("/compare_learning_curves", methods=["GET"])
def show_learning_curve_comparison():
    return render_template("learning_curve.html.j2")


@app.route("/runtimes", methods=["GET"])
def show_runtimes():
    return render_template("runtimes.html.j2")


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
    server.watch("*.py")
    server.watch("*.yaml")
    server.watch("*.j2")
    server.watch("*.css")
    server.watch("*.js")
    server.serve()
