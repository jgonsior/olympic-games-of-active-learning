from typing import List
from enum import Enum
from flask import Flask, render_template
from interactive_results_browser.csv_helper_functions import (
    get_exp_config_names,
    get_exp_grid_request_params,
)
from interactive_results_browser.visualizations import (
    vizualization_to_python_function_mapping,
)

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from misc.config import Config
from collections.abc import Iterable

import matplotlib.pyplot as plt

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)
app.debug = True


@app.route("/")
# @cache.cached(timeout=50)
def show_available_experiments():
    config = Config(no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": "only_random"})
    config_names = get_exp_config_names(config)
    return render_template("available_experiments.html.j2", config_names=config_names)


@app.route("/interactive/<string:experiment_name>", methods=["GET"])
def show_interactive_results(experiment_name: str):
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

    try:
        rendered_template = render_template(
            "show_interactive_metrics.html.j2",
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
    except Exception as ex:
        if hasattr(ex, "message"):
            ex = str(ex.message)
        else:
            ex = str(ex)

        import traceback

        ex = str(traceback.format_exc()) + "\n" + str(ex)
        """print("###" * 50)
        print(ex)
        print("###" * 50)
        print("\n" * 3)"""
        rendered_template = render_template(
            "error.html.j2",
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
            exception=ex,
        )

    return rendered_template


@app.route("/pcm/<string:experiment_name>", methods=["GET"])
def show_metrics_overview(experiment_name: str):
    config = Config(
        no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name},
    )

    exp_grid_request_params, full_exp_grid = get_exp_grid_request_params(
        experiment_name, config
    )

    rendered_template = render_template(
        "precomputed_metrics_overview.html.j2",
        experiment_name=experiment_name,
        config=config,
        full_exp_grid=full_exp_grid,
        isinstance=isinstance,
        Iterable=Iterable,
        Enum=Enum,
        int=int,
        str=str,
        exp_grid_request_params=exp_grid_request_params,
    )
    return rendered_template


@app.route("/sm/<string:experiment_name>/<string:metric_name>", methods=["GET"])
def single_metric(experiment_name: str):
    config = Config(
        no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name},
    )

    exp_grid_request_params, full_exp_grid = get_exp_grid_request_params(
        experiment_name, config
    )

    rendered_template = render_template(
        "precomputed_metrics_overview.html.j2",
        experiment_name=experiment_name,
        config=config,
        full_exp_grid=full_exp_grid,
        isinstance=isinstance,
        Iterable=Iterable,
        Enum=Enum,
        int=int,
        str=str,
        exp_grid_request_params=exp_grid_request_params,
    )
    return rendered_template
