from __future__ import annotations
import ast
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, List, TYPE_CHECKING


if TYPE_CHECKING:
    pass


class HARDEST_SAMPLES(Base_Computed_Metric):
    wrong_classified_counter: np.ndarray

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        # calculate the hardest samples for this dataset
        # check if calculation has already been done
        # if not: iterate over ALL strategies, and count per Y, how often the sample has been classified wrong
        # no matter the train/test/split
        hardest_samples_path = Path(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}/HARDEST_SAMPLES.csv.xz"
        )

        if not hardest_samples_path.exists():
            print("Computing ", hardest_samples_path)
            print("loading", EXP_DATASET)
            y_train_true: Dict[int, np.ndarray] = {}
            y_test_true: Dict[int, np.ndarray] = {}

            y = pd.read_csv(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.csv",
                usecols=["LABEL_TARGET"], engine="pyarrow"
            )["LABEL_TARGET"].to_numpy()

            self.wrong_classified_counter = np.zeros_like(y)

            _train_test_splits = pd.read_csv(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}", engine="pyarrow"
            )
            train_splits = {}
            test_splits = {}
            for train_test_split_ix, row in _train_test_splits.iterrows():
                train_set = ast.literal_eval(row["train"])
                test_set = ast.literal_eval(row["test"])

                y_train_true[train_test_split_ix] = y[train_set]
                y_test_true[train_test_split_ix] = y[test_set]

                train_splits[train_test_split_ix] = np.array(train_set)
                test_splits[train_test_split_ix] = np.array(test_set)

            for EXP_STRATEGY in self.config.EXP_GRID_STRATEGY:
                y_pred_train_path = Path(
                    self.config.OUTPUT_PATH
                    / EXP_STRATEGY.name
                    / EXP_DATASET.name
                    / "y_pred_train.csv.xz"
                )

                y_pred_test_path = Path(
                    self.config.OUTPUT_PATH
                    / EXP_STRATEGY.name
                    / EXP_DATASET.name
                    / "y_pred_test.csv.xz"
                )

                y_pred_train = pd.read_csv(y_pred_train_path, header=0, engine="pyarrow")
                y_pred = pd.read_csv(y_pred_test_path, header=0, engine="pyarrow").merge(
                    y_pred_train,
                    how="inner",
                    on="EXP_UNIQUE_ID",
                    suffixes=["_test", "_train"],
                )
                y_pred = y_pred.melt(
                    id_vars="EXP_UNIQUE_ID", var_name="train_or_test", value_name="y"
                )

                y_pred = y_pred.merge(
                    self.done_workload_df, how="left", on=["EXP_UNIQUE_ID"]
                )

                for _, row in y_pred.iterrows():
                    EXP_TRAIN_TEST_BUCKET_SIZE = row["EXP_TRAIN_TEST_BUCKET_SIZE"]
                    train_or_test = row["train_or_test"]
                    y_pred_single = np.array(ast.literal_eval(row["y"]))

                    if train_or_test.endswith("_train"):
                        y_true_single = y_train_true[EXP_TRAIN_TEST_BUCKET_SIZE]
                        global_indices = train_splits[EXP_TRAIN_TEST_BUCKET_SIZE]
                    else:
                        global_indices = test_splits[EXP_TRAIN_TEST_BUCKET_SIZE]
                        y_true_single = y_test_true[EXP_TRAIN_TEST_BUCKET_SIZE]

                    for wrong_ix in global_indices[
                        np.where(y_pred_single != y_true_single)
                    ]:
                        self.wrong_classified_counter[wrong_ix] += 1

            print(self.wrong_classified_counter)
            exit(-1)

    def computed_metric_appendix(self) -> str:
        return "mismatch_train_test"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, df.columns != "EXP_UNIQUE_ID"] = df.loc[
            :, df.columns != "EXP_UNIQUE_ID"
        ].apply(lambda x: [ast.literal_eval(iii) for iii in x], axis=0)
        return df

    def hardest_samples(
        self,
        row: pd.Series,
    ) -> pd.Series:
        print(row)
        exit(-1)
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
            new_metric_name="hardest_samples",
            apply_to_row=self.hardest_samples,
        )
