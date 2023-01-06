from __future__ import annotations
from abc import ABC
import csv
import gzip
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from misc.logging import log_it


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Base_Metric(ABC):
    metrics: List[str]
    metric_values: Dict[str, List[Any]]

    def __init__(self) -> None:
        self.metric_values = {}
        for metric in self.metrics:
            self.metric_values[metric] = []

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def save_metrics(self, al_experiment: AL_Experiment) -> None:
        for metric, values in self.metric_values.items():
            metric_result_file = Path(
                str(al_experiment.config.METRIC_RESULTS_FOLDER) + "/" + metric + ".csv"
            )

            values = {ix: v for ix, v in enumerate(values)}
            values["EXP_UNIQUE_ID"] = al_experiment.config.EXP_UNIQUE_ID

            with open(metric_result_file, "a") as f:
                w = csv.DictWriter(f, fieldnames=values.keys())

                if metric_result_file.stat().st_size == 0:
                    log_it("write headers first")
                    w.writeheader()
                w.writerow(values)
