from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING
from metrics.base_metric import Base_Metric


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


# TODO ersetzt pickled learner, ersetzt
class Predicted_Samples(Base_Metric):
    metrics = ["y_pred_train", "y_pred_test"]

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        y_pred_train = al_experiment.get_y_pred_train()
        y_pred_test = al_experiment.get_y_pred_test()
        self.metric_values["y_pred_train"].append(y_pred_train)
        self.metric_values["y_pred_test"].append(y_pred_test)
