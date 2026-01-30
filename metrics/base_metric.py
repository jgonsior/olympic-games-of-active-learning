"""Base metric class for OGAL experiments.

This module provides the abstract base class for all metrics computed
during Active Learning experiments. Metrics are recorded at each AL cycle
and saved to CSV files at experiment completion.

Key components:
    - Base_Metric: Abstract base class with hook methods for metric computation

Available hooks (called by AL_Experiment during each cycle):
    - pre_query_selection_hook: Before AL strategy selects samples
    - post_query_selection_hook: After selection, before training
    - pre_retraining_of_learner_hook: Before model.fit()
    - post_retraining_of_learner_hook: After model.fit()
    - save_metrics: At experiment end, writes all values to CSV

Concrete implementations:
    - Standard_ML_Metrics: Accuracy, F1, precision, recall
    - Selected_Indices: Track which samples were selected
    - Timing_Metrics: Query selection and retraining timing
    - Predicted_Samples: Model predictions per cycle

For more details, see docs/results_format.md
"""
from __future__ import annotations
from abc import ABC
import csv
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

from misc.logging import log_it


if TYPE_CHECKING:
    from framework_runners.base_runner import AL_Experiment


class Base_Metric(ABC):
    metrics: List[str]
    metric_values: Dict[str, List[Any]]

    padding_for_early_stopping = np.nan

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

            if len(values) < al_experiment.config.EXP_NUM_QUERIES:
                values += [
                    self.padding_for_early_stopping
                    for _ in range(len(values), al_experiment.config.EXP_NUM_QUERIES)
                ]

            values = {ix: v for ix, v in enumerate(values)}
            values["EXP_UNIQUE_ID"] = al_experiment.config.EXP_UNIQUE_ID

            with open(metric_result_file, "a") as f:
                w = csv.DictWriter(f, fieldnames=values.keys())

                if metric_result_file.stat().st_size == 0:
                    log_it("write headers first")
                    w.writeheader()
                w.writerow(values)
