from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    pass


class STANDARD_AUC(Base_Computed_Metric):
    metrics = [
        "accuracy",
        "macro_f1-score",
        "weighted_f1-score",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
    ]

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        del df["0"]
        return df

    # average over all iterations (0-20)
    def full_auc(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        return row.sum() / row.notna().sum(0)

    # average of first five iterations (1-5)
    def ramp_up_quality(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row[0:5]
        return row.sum() / row.notna().sum(0)

    # average of middle iterations (5-15)
    def middle_quality(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row[5:15]
        return row.sum() / row.notna().sum(0)

    # average of last five iterations (15-20)
    def end_quality(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row[15:20]
        return row.sum() / row.notna().sum(0)

    # see trittenbach 2020 paper: LS(k) = (QR(end-k, end)/k)/(QR(init,end)/|L^end \without L^init)
    # intuition: we calculate the average improvement of the last five AL cycles, and divide by the average attribution of all cycles
    # if the last five steps are still improving -> 20 iterations is not enough apparently, otherwise, it is
    def learning_stabilty(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        diff = row.diff()[1:]
        overall_average_improvement = diff.mean()
        last_improvement = diff[-5:].mean()

        if overall_average_improvement <= 0:
            ls = 0
        else:
            ls = last_improvement / overall_average_improvement

        return ls

    def compute(self) -> None:
        for metric in self.metrics:
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="ramp_up_quality_" + metric,
                apply_to_row=self.ramp_up_quality,
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="middle_quality_" + metric,
                apply_to_row=self.middle_quality,
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="end_quality_" + metric,
                apply_to_row=self.end_quality,
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="learning_stabilty_" + metric,
                apply_to_row=self.learning_stabilty,
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_" + metric,
                apply_to_row=self.full_auc,
            )
