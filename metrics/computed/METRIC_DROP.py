from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, List, Tuple
import pandas as pd
from datasets import DATASET

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

    def biggest_drop(self, row: pd.Series, EXP_DATASET: DATASET) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        biggest_drop = 0

        for ix, (v0, v1) in enumerate(zip(row[0:-1], row[1:])):
            diff = v0 - v1
            if diff > biggest_drop:
                biggest_drop = diff
            row.iloc[ix] = -biggest_drop
        row = row[:-1]
        return row

    def nr_decreasing_al_cycles(
        self, row: pd.Series, EXP_DATASET: DATASET
    ) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        nr_drops = 0

        for ix, (v0, v1) in enumerate(zip(row[0:-1], row[1:])):
            if v1 < v0:
                nr_drops += 1
            row.iloc[ix] = -nr_drops
        row = row[:-1]
        return row

    def get_all_metric_jobs(self) -> List[Tuple[Callable, List[Any]]]:
        results = []
        for basic_metric in self.metrics:
            results = [
                *results,
                *self._compute_single_metric_jobs(
                    existing_metric_names=[basic_metric],
                    new_metric_name="biggest_drop_per_" + basic_metric,
                    apply_to_row=self.biggest_drop,
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[basic_metric],
                    new_metric_name="nr_decreasing_al_cycles_per_" + basic_metric,
                    apply_to_row=self.nr_decreasing_al_cycles,
                ),
            ]
        return results
