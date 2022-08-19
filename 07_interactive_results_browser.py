from flask import Flask, render_template
from flask import request
from interactive_results_browser.csv_helper_functions import get_exp_config_names
from livereload import Server
from misc.config import Config

app = Flask(
    __name__,
    template_folder="interactive_results_browser/templates",
    static_folder="interactive_results_browser/static",
)
app.debug = True

config = Config(no_cli_args={"WORKER_INDEX": None})


@app.route("/")
def show_available_experiments():
    # parse exp_config.yaml
    # display links for each of the available workloads
    # all of them go into
    config_names = get_exp_config_names(config)
    return render_template("index.html")
    return "<br>".join(config_names)


@app.route("/workload/<string:experiment_name>", methods=["GET"])
def show_open_done_workload(experiment_name: str):
    return "<p>Hello, Worldl!</p>"


@app.route("/dataset/<int:dataset_id>", methods=["GET"])
def show_dataset_overview(dataset_id: int):
    return f"{dataset_id}"


@app.route("/strategy/<int:strategy_id>", methods=["GET"])
def show_strategy_overview(strategy_id: int):
    return f"{strategy_id}"


@app.route("/compare_learning_curves", methods=["GET"])
def show_learning_curve_comparison():

    return f"not implemented yet"


if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.serve()
