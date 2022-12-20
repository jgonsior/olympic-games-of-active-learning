from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING
from metrics.base_metric import Base_Metric
import time

if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Timing_Metrics(Base_Metric):
    metrics = ["learner_training_time", "query_selection_time"]

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        self.learner_training_start_time = time.process_time()

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        self.metric_values["learner_training_time"].append(
            time.process_time() - self.learner_training_start_time
        )

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        self.query_selection_start_time = time.process_time()

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        self.metric_values["query_selection_time"].append(
            time.process_time() - self.query_selection_start_time
        )
