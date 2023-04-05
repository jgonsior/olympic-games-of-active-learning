from __future__ import annotations
import glob
from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    pass


class STANDARD_AUC(Base_Computed_Metric):
    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        if "0" in df.columns:
            del df["0"]
        df = self._parse_using_ast_literal_eval(df, calculate_mean_too=True)
        return df

    def range_auc(self, row: pd.Series, range_start: int, range_end: int) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row[range_start:range_end]
        return row.sum() / row.notna().sum(0)

    # see trittenbach 2020 paper: LS(k) = (QR(end-k, end)/k)/(QR(init,end)/|L^end \without L^init)
    # intuition: we calculate the average improvement of the last five AL cycles, and divide by the average attribution of all cycles
    # if the last five steps are still improving -> 20 iterations is not enough apparently, otherwise, it is
    def learning_stability(self, row: pd.Series) -> pd.Series:
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
        ]

        for metric in all_existing_metric_names:
            """self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_0_5_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 0, "range_end": 5},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_5_15_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 5, "range_end": 15},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_15_20_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 15, "range_end": 20},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_0_7_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 0, "range_end": 7},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_7_14_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 7, "range_end": 14},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_14_20_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 14, "range_end": 20},
            )
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="learning_stability_" + metric,
                apply_to_row=self.learning_stability,
            )"""
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name="auc_full_" + metric,
                apply_to_row=self.range_auc,
                additional_apply_to_row_kwargs={"range_start": 0, "range_end": -1},
            )
