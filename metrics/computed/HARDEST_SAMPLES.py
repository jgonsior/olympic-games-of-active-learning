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
        hardest_samples_path = Path(
            f"{self.config.OUTPUT_PATH}/_hardest_samples/{EXP_DATASET.name}_HARDEST_SAMPLES.npz"
        )

        if not hardest_samples_path.exists():
            print("Computing ", hardest_samples_path)
            print("loading", EXP_DATASET)
            y_train_true: Dict[int, np.ndarray] = {}
            y_test_true: Dict[int, np.ndarray] = {}

            y = pd.read_csv(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.csv",
                usecols=["LABEL_TARGET"],
            )["LABEL_TARGET"].to_numpy()

            self.wrong_classified_counter = np.zeros_like(y)

            _train_test_splits = pd.read_csv(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}"
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

                if not y_pred_train_path.exists():
                    continue

                y_pred_train = pd.read_csv(y_pred_train_path, header=0)
                y_pred = pd.read_csv(y_pred_test_path, header=0).merge(
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

            hardest_samples_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                hardest_samples_path,
                wrong_classified_counter=self.wrong_classified_counter,
            )
        else:
            print("Loading ", hardest_samples_path)

            self.wrong_classified_counter = np.load(hardest_samples_path)[
                "wrong_classified_counter"
            ]

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        del df["0"]
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")
        df["selected_indices"] = df[column_names_which_are_al_cycles].apply(
            lambda x: ast.literal_eval(
                "[" + ",".join(x).replace("[", "").replace("]", "") + "]"
            ),
            axis=1,
        )
        for c in column_names_which_are_al_cycles:
            del df[c]

        return df

    def hardest_samples(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return np.sum(self.wrong_classified_counter[row["selected_indices"]])

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="hardest_samples",
            apply_to_row=self.hardest_samples,
        )
