from __future__ import annotations
from typing import TYPE_CHECKING

from sklearn.metrics import classification_report
from metrics.base_metric import Base_Metric


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Standard_ML_Metrics(Base_Metric):
    metrics = [
        "accuracy",
        "macro_f1-score",
        "weighted_f1-score",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
    ]

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        classification_report_results = classification_report(
            y_true=al_experiment.Y[al_experiment.test_idx],
            y_pred=al_experiment.get_y_pred_test(),
            output_dict=True,
            zero_division=0,
        )

        self.metric_values["accuracy"].append(classification_report_results["accuracy"])
        self.metric_values["macro_f1-score"].append(
            classification_report_results["macro avg"]["f1-score"]
        )
        self.metric_values["macro_f1-score"].append(
            classification_report_results["macro avg"]["f1-score"]
        )
        self.metric_values["macro_precision"].append(
            classification_report_results["macro avg"]["precision"]
        )
        self.metric_values["macro_recall"].append(
            classification_report_results["macro avg"]["recall"]
        )
        self.metric_values["weighted_f1-score"].append(
            classification_report_results["weighted avg"]["f1-score"]
        )
        self.metric_values["weighted_precision"].append(
            classification_report_results["weighted avg"]["precision"]
        )
        self.metric_values["weighted_recall"].append(
            classification_report_results["weighted avg"]["recall"]
        )
