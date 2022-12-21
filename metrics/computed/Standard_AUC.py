from __future__ import annotations
import pandas as pd

from metrics.computed.base_computed_metric import Base_Computed_Metric


class Standard_AUC(Base_Computed_Metric):
    metrics = [
        "accuracy",
        "macro_f1-score",
        "weighted_f1-score",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
    ]

    computed_metric_appendix = "_auc"

    def apply_to_row(self, row: pd.Series) -> pd.Series:
        return row.sum() / len(row)
