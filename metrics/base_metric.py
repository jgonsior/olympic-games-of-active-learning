from abc import ABC, abstractmethod
from typing import List

from framework_runners.base_runner import AL_Experiment


class Base_Metric(ABC):
    metrics: List[str] = []
    metric_values: Dict[str, List[Any]] = {}

    def __init__(self) -> None:
        for metric in self.metrics:
            self.metric_values[metric] = []

    @abstractmethod
    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def save_metrics(self) -> None:
        for metric in self.metrics:
            # check if csv file exists
            # if yes, append our values
            # final csv file as 20 columns, each per timestamp, and each column gets the value of this single metric
            ...
