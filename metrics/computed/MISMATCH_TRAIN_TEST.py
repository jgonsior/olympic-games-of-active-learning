from __future__ import annotations
import ast
import numpy as np
import pandas as pd
from datasets import DATASET

import dask.dataframe as dd
from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY


class MISMATCH_TRAIN_TEST(Base_Computed_Metric):
    y_train_true: Dict[DATASET, Dict[int, np.ndarray]] = {}
    y_test_true: Dict[DATASET, Dict[int, np.ndarray]] = {}

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        _train_test_splits = dd.read_parquet(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}.parquet",
        ).compute()
        y = (
            dd.read_parquet(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.parquet",
                usecols=["LABEL_TARGET"],
            )["LABEL_TARGET"]
            .compute()
            .to_numpy()
        )

        for train_test_split_ix, row in _train_test_splits.iterrows():
            train_set = ast.literal_eval(row["train"])
            test_set = ast.literal_eval(row["test"])

            if EXP_DATASET not in self.y_train_true.keys():
                self.y_train_true[EXP_DATASET] = {}
                self.y_test_true[EXP_DATASET] = {}

            self.y_train_true[EXP_DATASET][train_test_split_ix] = y[train_set]
            self.y_test_true[EXP_DATASET][train_test_split_ix] = y[test_set]

        print("done loading")

    def computed_metric_appendix(self) -> str:
        return "mismatch_train_test"

    def mismatch_train_test(self, row: pd.Series, EXP_DATASET: DATASET) -> pd.Series:
        unique_id = row["EXP_UNIQUE_ID"]
        train_test_split_number = self.done_workload_df.loc[
            self.done_workload_df["EXP_UNIQUE_ID"] == unique_id
        ]["EXP_TRAIN_TEST_BUCKET_SIZE"].to_list()[0]
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        # calculate how many y's in train richtig vorhergesagt
        # calculate how many y's in test richtig vorhergesagt
        # differenz ist dann das ergebnis
        amount_of_al_iterations = int(len(row) / 2)

        results: Dict[int, int] = {}
        for al_iteration in range(0, amount_of_al_iterations):
            y_pred_train = row[f"{al_iteration}_x"]
            y_pred_test = row[f"{al_iteration}_y"]

            if len(y_pred_train) == 0:
                # we don't have so many labeled data for so many iterations
                results[al_iteration] = np.nan
            else:
                amount_of_correct_predicted_train = np.sum(
                    y_pred_train
                    == self.y_train_true[EXP_DATASET][train_test_split_number]
                ) / len(y_pred_train)
                amount_of_correct_predicted_test = np.sum(
                    y_pred_test
                    == self.y_test_true[EXP_DATASET][train_test_split_number]
                ) / len(y_pred_test)

                results[al_iteration] = (
                    amount_of_correct_predicted_test / amount_of_correct_predicted_train
                )

        return pd.Series(results)

    def compute_metrics(self, exp_dataset: DATASET, exp_strategy: AL_STRATEGY):
        self._per_dataset_hook(exp_dataset)

        self._compute_single_metric_jobs(
            existing_metric_names=["y_pred_train", "y_pred_test"],
            new_metric_name="mismatch_train_test",
            apply_to_row=self.mismatch_train_test,
            exp_dataset=exp_dataset,
            exp_strategy=exp_strategy,
        )
