from __future__ import annotations
import ast
import itertools
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, List, TYPE_CHECKING, Literal
from scipy.spatial import distance

if TYPE_CHECKING:
    pass


class CLASS_DISTRIBUTIONS(Base_Computed_Metric):
    y_true: np.ndarray
    true_distribution: Dict[int, Dict[int, int]] = {}
    classes: List[int]

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        _train_test_splits = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}", engine="pyarrow"
        )
        self.y_true = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.csv",
            usecols=["LABEL_TARGET"], engine="pyarrow"
        )["LABEL_TARGET"].to_numpy()
        self.classes = np.unique(self.y_true).tolist()
        for train_test_split_ix, row in _train_test_splits.iterrows():
            train_set = ast.literal_eval(row["train"])

            _, self.true_distribution[train_test_split_ix] = np.unique(
                self.y_true[train_set], return_counts=True
            )  # type:ignore
            self.true_distribution[train_test_split_ix] = [
                x / np.sum(self.true_distribution[train_test_split_ix])
                for x in self.true_distribution[train_test_split_ix]
            ]

        print("done loading")

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

    def _class_distributions(
        self, row: pd.Series, metric: Literal["manhattan", "chebyshev"]
    ) -> pd.Series:
        unique_id = row["EXP_UNIQUE_ID"]
        train_test_split_number = self.done_workload_df.loc[
            self.done_workload_df["EXP_UNIQUE_ID"] == unique_id
        ]["EXP_TRAIN_TEST_BUCKET_SIZE"].to_list()[0]

        selected_indices = row.loc[row.index != "EXP_UNIQUE_ID"]["selected_indices"]
        selected_classes = self.y_true[selected_indices]
        k, class_distributions = np.unique(selected_classes, return_counts=True)

        if len(k) != len(self.true_distribution[train_test_split_number]):
            new_class_distributions = [0 for _ in self.classes]
            for key, value in zip(k, class_distributions):
                new_class_distributions[self.classes.index(key)] = value
            class_distributions = new_class_distributions

        class_distributions = [
            x / sum(class_distributions) for x in class_distributions
        ]

        true_counts = self.true_distribution[train_test_split_number]
        pred_counts = class_distributions

        if metric == "manhattan":
            total_diff = 0
            for true_count, pred_count in zip(true_counts, pred_counts):
                # distance metrics: https://stats.stackexchange.com/a/151362
                total_diff += abs(true_count - pred_count)
            result = total_diff
        elif metric == "chebyshev":
            result = distance.chebyshev(true_counts, pred_counts)

        return result

    def class_distributions_manhattan(self, row: pd.Series) -> pd.Series:
        return self._class_distributions(row, metric="manhattan")

    def unifclass_distributions_chebyshev(self, row: pd.Series) -> pd.Series:
        return self._class_distributions(row, metric="chebyshev")

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_chebyshev",
            apply_to_row=self.unifclass_distributions_chebyshev,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_manhattan",
            apply_to_row=self.class_distributions_manhattan,
        )
