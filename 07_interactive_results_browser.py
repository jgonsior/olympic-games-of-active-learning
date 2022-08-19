from flask import Flask
from flask import request

from misc.config import Config


app = Flask(__name__)


config = Config(no_cli_args={"WORKER_INDEX": None})


@app.route("/")
def show_available_experiments():
    # parse exp_config.yaml
    # display links for each of the available workloads
    # all of them go into

    return "nothing to see here yet"


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
