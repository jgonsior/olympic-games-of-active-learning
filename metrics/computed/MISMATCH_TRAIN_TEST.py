from __future__ import annotations
import ast
import itertools
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, List, TYPE_CHECKING


if TYPE_CHECKING:
    pass


class MISMATCH_TRAIN_TEST(Base_Computed_Metric):
    y_train_true: Dict[int, np.ndarray] = {}
    y_test_true: Dict[int, np.ndarray] = {}

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        _train_test_splits = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}", engine="pyarrow"
        )
        y = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.csv", engine="pyarrow"
            usecols=["LABEL_TARGET"],
        )["LABEL_TARGET"].to_numpy()

        for train_test_split_ix, row in _train_test_splits.iterrows():
            train_set = ast.literal_eval(row["train"])
            test_set = ast.literal_eval(row["test"])

            self.y_train_true[train_test_split_ix] = y[train_set]
            self.y_test_true[train_test_split_ix] = y[test_set]

        print("done loading")

    def computed_metric_appendix(self) -> str:
        return "mismatch_train_test"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, df.columns != "EXP_UNIQUE_ID"] = df.loc[
            :, df.columns != "EXP_UNIQUE_ID"
        ].apply(lambda x: [ast.literal_eval(iii) for iii in x], axis=0)
        return df

    def mismatch_train_test(
        self,
        row: pd.Series,
    ) -> pd.Series:
        unique_id = row["EXP_UNIQUE_ID"]
        train_test_split_number = self.done_workload_df.loc[
            self.done_workload_df["EXP_UNIQUE_ID"] == unique_id
        ]["EXP_TRAIN_TEST_BUCKET_SIZE"].to_list()[0]

        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        # calculate how many y's in train richtig vorhergesagt
        # calculate how many y's in test richtig vorhergesagt
        # differenz ist dann das ergebnis
        amount_of_al_iterations = int(len(row) / 2)

        results = 0
        for al_iteration in range(1, amount_of_al_iterations):
            y_pred_train = row[f"{al_iteration}_x"]
            y_pred_test = row[f"{al_iteration}_y"]
            amount_of_correct_predicted_train = np.sum(
                y_pred_train == self.y_train_true[train_test_split_number]
            ) / len(y_pred_train)
            amount_of_correct_predicted_test = np.sum(
                y_pred_test == self.y_test_true[train_test_split_number]
            ) / len(y_pred_test)
            results += (
                amount_of_correct_predicted_test - amount_of_correct_predicted_train
            )
        return results

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["y_pred_train", "y_pred_test"],
            new_metric_name="mismatch_train_test",
            apply_to_row=self.mismatch_train_test,
        )
