from __future__ import annotations
import glob
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    pass


class TIMELAG_METRIC(Base_Computed_Metric):
    # takes in standard metric, and calculates the timelag difference -> removes trend from learning curve time series

    def time_lag(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row.diff()
        row.drop(labels=["0"], inplace=True)
        return row

    def compute(self) -> None:
        all_existing_metric_names = set(
            [
                Path(a).name
                for a in glob.glob(str(self.config.OUTPUT_PATH / "*/*/*.csv.xz"))
            ]
        )
        all_existing_metric_names = [
            a.split(".")[0]
            for a in all_existing_metric_names
            if not a.startswith("auc_")
            and not a.startswith("learning_stability_")
            and not a.startswith("pickled_learner_model")
            and not a.endswith("_time_lag.csv.xz")
            and not a.endswith("y_pred_test.csv.xz")
            and not a.endswith("y_pred_train.csv.xz")
            and not a.endswith("selected_indices.csv.xz")
        ]

        for metric in all_existing_metric_names:
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name=metric + "_time_lag",
                apply_to_row=self.time_lag,
            )
