import enum
from pathlib import Path
from flask import Flask, render_template
from datasets import DATASET
from interactive_results_browser.csv_helper_functions import (
    create_open_done_workload_table,
    get_exp_config_names,
    load_workload_csv_files,
    get_exp_grid_request_params,
)
from livereload import Server
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


@app.route("/result/<string:experiment_name>", methods=["GET"])
def show_results(experiment_name: str):
    config = Config(
        no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name},
    )

    exp_grid_request_params, full_exp_grid = get_exp_grid_request_params(
        experiment_name, config
    )

    for viz in exp_grid_request_params["VISUALIZATIONS"]:
        print(viz)

    visualizations_and_tables = []

    full_workload, open_jobs, done_jobs = load_workload_csv_files(config)

    open_done_df = create_open_done_workload_table(
        full_workload,
        open_jobs,
        done_jobs,
        config,
        exp_grid_request_params,
    )
    print(exp_grid_request_params)

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
        exp_grid_request_params=exp_grid_request_params,
        full_exp_grid=full_exp_grid,
        type=type,
        tuple=tuple,
        enum=enum.Enum,
        int=int,
        config=config,
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
