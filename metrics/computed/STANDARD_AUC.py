from __future__ import annotations
from typing import TYPE_CHECKING
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY


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

    def computed_metric_appendix(self) -> str:
        return "auc"

    def apply_to_row(
        self, row: pd.Series, EXP_STRATEGY: AL_STRATEGY, EXP_DATASET: DATASET
    ) -> pd.Series:
        return row.sum() / len(row)
