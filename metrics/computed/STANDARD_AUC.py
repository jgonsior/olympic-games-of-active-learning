from __future__ import annotations
import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
import pandas as pd

from datasets import DATASET
from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    from misc.config import Config


class STANDARD_AUC(Base_Computed_Metric):
    _dataset_dependend_thresholds_df: pd.DataFrame

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._dataset_dependend_thresholds_df = pd.read_csv(
            self.config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH
        )

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        if "0" in df.columns:
            del df["0"]

        if isinstance(df["1"].dtypes, object):
            df = self._parse_using_ast_literal_eval(df, calculate_mean_too=True)
        return df

    def range_auc(
        self, row: pd.Series, range_start: int, range_end: int, EXP_DATASET: DATASET
    ) -> pd.Series:
        if range_start == "pre_computed":
            range_start = self._dataset_dependend_thresholds_df[
                self._dataset_dependend_thresholds_df["EXP_UNIQUE_ID"]
                == row["EXP_UNIQUE_ID"]
            ]["cutoff_value"]

        if range_end == "pre_computed":
            range_start = self._dataset_dependend_thresholds_df[
                self._dataset_dependend_thresholds_df["EXP_UNIQUE_ID"]
                == row["EXP_UNIQUE_ID"]
            ]["cutoff_value"]
        # TODO: check if indexing with -1 and -5 etc. works
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row[range_start:range_end]

        return row.sum() / row.notna().sum(0)

    # see trittenbach 2020 paper: LS(k) = (QR(end-k, end)/k)/(QR(init,end)/|L^end \without L^init)
    # intuition: we calculate the average improvement of the last five AL cycles, and divide by the average attribution of all cycles
    # if the last five steps are still improving -> 20 iterations is not enough apparently, otherwise, it is
    def learning_stability(
        self, row: pd.Series, EXP_DATASET: DATASET, time_range: int
    ) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        diff = row.diff()[1:]
        overall_average_improvement = diff.mean()
        last_improvement = diff[-time_range:].mean()

        if overall_average_improvement <= 0:
            ls = 0
        else:
            ls = last_improvement / overall_average_improvement

        return ls

    def get_all_metric_jobs(self) -> List[Tuple[Callable, List[Any]]]:
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
            and not a.startswith("y_pred_train")
            and not a.startswith("y_pred_test")
            and not a.startswith("selected_indices")
        ]
        results = []

        for metric in all_existing_metric_names:
            results = [
                *results,
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="final_value" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={"range_start": -1, "range_end": -1},
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="first_5" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={"range_start": 0, "range_end": 5},
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="last_5" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={"range_start": -5, "range_end": -1},
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="ramp_up_auc_" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={
                        "range_start": 0,
                        "range_end": "pre_calculated",
                    },
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="plateau_auc_" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={
                        "range_start": "pre_calculated",
                        "range_end": -1,
                    },
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="learning_stability_5_" + metric,
                    apply_to_row=self.learning_stability,
                    additional_apply_to_row_kwargs={"time_range": 5},
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="learning_stability_10_" + metric,
                    apply_to_row=self.learning_stability,
                    additional_apply_to_row_kwargs={"time_range": 10},
                ),
                *self._compute_single_metric_jobs(
                    existing_metric_names=[metric],
                    new_metric_name="full_auc_" + metric,
                    apply_to_row=self.range_auc,
                    additional_apply_to_row_kwargs={"range_start": 0, "range_end": -1},
                ),
            ]
        return results
