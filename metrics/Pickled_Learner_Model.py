from __future__ import annotations
import pickle
from typing import TYPE_CHECKING
from metrics.base_metric import Base_Metric


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Pickled_Learner_Model(Base_Metric):
    metrics = ["pickled_learner_model"]

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        self.metric_values["pickled_learner_model"].append(
            pickle.dumps(al_experiment.model, protocol=5)
        )
