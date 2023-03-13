from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    pass


class METRIC_DROP(Base_Computed_Metric):
    metrics = [
        "accuracy",
        "macro_f1-score",
        "weighted_f1-score",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
    ]

    def computed_metric_appendix(self) -> str:
        return "auc"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def biggest_drop(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        biggest_drop = 0

        for ix, (v0, v1) in enumerate(zip(row[0:-1], row[1:])):
            diff = v0 - v1
            if diff > biggest_drop:
                biggest_drop = diff
            row[ix] = -biggest_drop
        row = row[:-1]
        return row

    def nr_decreasing_al_cycles(self, row: pd.Series) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        nr_drops = 0

        for ix, (v0, v1) in enumerate(zip(row[0:-1], row[1:])):
            if v1 < v0:
                nr_drops += 1
            row[ix] = -nr_drops
        row = row[:-1]
        return row

    def compute(self) -> None:
        for basic_metric in self.metrics:
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[basic_metric],
                new_metric_name="biggest_drop_per_" + basic_metric,
                apply_to_row=self.biggest_drop,
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[basic_metric],
                new_metric_name="nr_decreasing_al_cycles_per_" + basic_metric,
                apply_to_row=self.nr_decreasing_al_cycles,
            )
