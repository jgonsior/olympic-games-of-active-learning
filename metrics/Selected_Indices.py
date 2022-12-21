from __future__ import annotations
from typing import TYPE_CHECKING
from metrics.base_metric import Base_Metric


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Selected_Indices(Base_Metric):
    metrics = ["selected_indices"]

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        self.metric_values["selected_indices"].append(al_experiment.select_ind)
