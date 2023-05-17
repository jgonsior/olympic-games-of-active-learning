from __future__ import annotations
import math
import ast
import itertools
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    pass


class DISTANCE_METRICS(Base_Computed_Metric):
    metrics = ["avg_dist_batch", "avg_dist_labeled", "avg_dist_unlabeled"]

    _precomputed_distances: np.ndarray
    _train_test_splits: pd.DataFrame

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        self._precomputed_distances = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_DISTANCES_APPENDIX}",
        ).to_numpy()
        self._train_test_splits = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}",
            delimiter=",",
            index_col=False,
        )

        print("done loading")

    def computed_metric_appendix(self) -> str:
        return "dist"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")

        df[column_names_which_are_al_cycles] = df[
            column_names_which_are_al_cycles
        ].applymap(lambda x: "[]" if pd.isna(x) else x)
        df.loc[:, column_names_which_are_al_cycles] = df.loc[
            :, column_names_which_are_al_cycles
        ].apply(lambda x: [ast.literal_eval(iii) for iii in x], axis=0)
        return df

    def avg_dist_batch(self, row: pd.Series,) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]

        for ix, x in row.items():
            if ix == 0:
                continue
            distances = []
            for s1, s2 in itertools.combinations(x, 2):
                distances.append(self._precomputed_distances[s1][s2])

            if len(distances) == 0:
                row[ix] = 0
            else:
                row[ix] = sum(distances) / len(distances)
        return row

    def avg_dist_labeled(self, row: pd.Series,) -> pd.Series:
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        labeled_so_far = []
        for ix, x in row.items():
            if ix == 0:
                labeled_so_far = x
            distances = []

            for s1 in x:
                for s2 in labeled_so_far:
                    distances.append(self._precomputed_distances[s1][s2])

            labeled_so_far += x

            if len(distances) == 0:
                row[ix] = 0
            else:
                row[ix] = sum(distances) / len(distances)
        return row

    def _get_train_set(self, unique_id: int) -> List[int]:
        details = self.done_workload_df.loc[
            self.done_workload_df["EXP_UNIQUE_ID"] == unique_id
        ]
        train_set = ast.literal_eval(
            self._train_test_splits["train"].iloc[
                details["EXP_TRAIN_TEST_BUCKET_SIZE"].to_list()[0]
            ]
        )
        return train_set

    def avg_dist_unlabeled(self, row: pd.Series,) -> pd.Series:
        unique_id = row["EXP_UNIQUE_ID"]
        row = row.loc[row.index != "EXP_UNIQUE_ID"]
        train_set = self._get_train_set(unique_id)
        unlabeled_so_far = set(train_set)
        for ix, x in row.items():
            if ix == 0:
                for s1 in x:
                    unlabeled_so_far.remove(s1)

            distances = []

            for s1 in x:
                for s2 in unlabeled_so_far:
                    distances.append(self._precomputed_distances[s1][s2])

            for s1 in x:
                unlabeled_so_far.remove(s1)

            if len(distances) == 0:
                row[ix] = 0
            else:
                row[ix] = sum(distances) / len(distances)
        return row

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="avg_dist_batch",
            apply_to_row=self.avg_dist_batch,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="avg_dist_labeled",
            apply_to_row=self.avg_dist_labeled,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="avg_dist_unlabeled",
            apply_to_row=self.avg_dist_unlabeled,
        )
