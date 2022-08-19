from flask import Flask, render_template
from flask import request
from interactive_results_browser.csv_helper_functions import (
    create_open_done_workload_table,
    get_exp_config_names,
    load_workload_csv_files,
)
from livereload import Server
from misc.config import Config

app = Flask(
    __name__,
    template_folder="interactive_results_browser/templates",
    static_folder="interactive_results_browser/static",
)
app.debug = True


@app.route("/")
def show_available_experiments():

    config = Config(no_cli_args={"WORKER_INDEX": None})
    # parse exp_config.yaml
    # display links for each of the available workloads
    # all of them go into
    config_names = get_exp_config_names(config)
    return render_template("available_experiments.html.j2", config_names=config_names)


@app.route("/workload/<string:experiment_name>", methods=["GET"])
def show_open_done_workload(experiment_name: str):
    config = Config(no_cli_args={"WORKER_INDEX": None, "EXP_TITLE": experiment_name})

    full_workload, open_jobs, done_jobs = load_workload_csv_files(config)
    open_done_df = create_open_done_workload_table(
        full_workload,
        open_jobs,
        done_jobs,
        config,
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
    )


@app.route("/dataset/<int:dataset_id>", methods=["GET"])
def show_dataset_overview(dataset_id: int):
    return render_template("dataset_overview.html.j2")


@app.route("/strategy/<int:strategy_id>", methods=["GET"])
def show_strategy_overview(strategy_id: int):
    return render_template("strategy_overview.html.j2")


@app.route("/compare_learning_curves", methods=["GET"])
def show_learning_curve_comparison():
    return render_template("learning_curve.html.j2")


if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.serve()
