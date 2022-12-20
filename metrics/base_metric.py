from __future__ import annotations
from abc import ABC, abstractmethod
import csv
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
        print(self.metric_values)
        for metric, values in self.metric_values.items():
            # print(metric)
            # print(values)
            # check if csv file exists
            # if yes, append our values
            # final csv file as 20 columns, each per timestamp, and each column gets the value of this single metric

            # eine "done workload" pro strategie/datensatz ordner --> die werden dann zusammengemerged!

            with open(al_experiment.config.DONE_WORKLOAD_PATH, "a") as f:
                w = csv.DictWriter(f, fieldnames=[a for a in range(0, len(values))])

                if al_experiment.config.DONE_WORKLOAD_PATH.stat().st_size == 0:
                    log_it("write headers first")
                    w.writeheader()
                w.writerow(values)
