from __future__ import annotations
import glob
from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY


class TIMELAG_METRIC(Base_Computed_Metric):
    # takes in standard metric, and calculates the timelag difference -> removes trend from learning curve time series
    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        # check if lists are existent in df, if so, calculate mean etc.
        if isinstance(df["0"].dtypes, object):
            df = self._parse_using_ast_literal_eval(df, calculate_mean_too=True)

        return df

    def time_lag(self, row: pd.Series, EXP_DATASET: DATASET) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        row = row.diff()
        row.drop(labels=["0"], inplace=True)
        return row

    def compute_metrics(self, exp_dataset: DATASET, exp_strategy: AL_STRATEGY):
        all_existing_metric_names = set(
            [
                Path(a).name
                for a in glob.glob(str(self.config.OUTPUT_PATH / "*/*/*.csv.xz"))
            ]
        )
        all_existing_metric_names = [
            a.split(".")[0]
            for a in all_existing_metric_names
            if not a.startswith("ramp_up_auc_")
            and not a.startswith("plateau_auc_")
            and not a.startswith("final_value_")
            and not a.startswith("first_5_")
            and not a.startswith("last_5_")
            and not a.startswith("learning_stability_5_")
            and not a.startswith("learning_stability_10_")
            and not a.startswith("full_auc_")
            and not a.startswith("pickled_learner_model")
            and not a.startswith("y_pred_train")
            and not a.startswith("y_pred_test")
            and not a.startswith("selected_indices")
            and not a.endswith("_time_lag.csv.xz")
            #  and not "accuracy" in a
            #  and not "weighted_f1-score" in a
            and not "macro_precision" in a
            and not "weighted_precision" in a
            and not "macro_recall" in a
            and not "weighted_recall" in a
        ]

        for metric in all_existing_metric_names:
            self._compute_single_metric_jobs(
                existing_metric_names=[metric],
                new_metric_name=metric + "_time_lag",
                apply_to_row=self.time_lag,
                exp_dataset=exp_dataset,
                exp_strategy=exp_strategy,
            )
