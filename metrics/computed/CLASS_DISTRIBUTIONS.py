from __future__ import annotations
import ast
import numpy as np
import pandas as pd
from datasets import DATASET

import dask.dataframe as dd
from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, List, TYPE_CHECKING, Literal
from scipy.spatial import distance

if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY


class CLASS_DISTRIBUTIONS(Base_Computed_Metric):
    y_true: Dict[DATASET, np.ndarray] = {}
    true_distribution: Dict[DATASET, Dict[int, Dict[int, int]]] = {}
    classes: Dict[DATASET, List[int]] = {}

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        _train_test_splits = dd.read_parquet(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}.parquet"
        ).compute()
        self.y_true[EXP_DATASET] = (
            dd.read_parquet(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.parquet",
                usecols=["LABEL_TARGET"],
            )["LABEL_TARGET"]
            .compute()
            .to_numpy()
        )

        self.classes[EXP_DATASET] = np.unique(self.y_true[EXP_DATASET]).tolist()
        for train_test_split_ix, row in _train_test_splits.iterrows():
            train_set = ast.literal_eval(row["train"])

            if EXP_DATASET not in self.true_distribution.keys():
                self.true_distribution[EXP_DATASET] = {}

            _, self.true_distribution[EXP_DATASET][train_test_split_ix] = np.unique(
                self.y_true[EXP_DATASET][train_set], return_counts=True
            )  # type:ignore
            self.true_distribution[EXP_DATASET][train_test_split_ix] = [
                x / np.sum(self.true_distribution[EXP_DATASET][train_test_split_ix])
                for x in self.true_distribution[EXP_DATASET][train_test_split_ix]
            ]

        print("done loading")

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._convert_selected_indices_to_ast(df, merge=False)
        return df

    # two variants
    # variant a: per batch
    # variant b: for all selected indices up until this point
    def _class_distributions(
        self,
        row: pd.Series,
        batch: bool,
        metric: Literal["manhattan", "chebyshev"],
        EXP_DATASET: DATASET,
    ) -> pd.Series:
        unique_id = row["EXP_UNIQUE_ID"]
        train_test_split_number = self.done_workload_df.loc[
            self.done_workload_df["EXP_UNIQUE_ID"] == unique_id
        ]["EXP_TRAIN_TEST_BUCKET_SIZE"].to_list()[0]

        col_names_without_exp_unique_id = [i for i in row.index if i != "EXP_UNIQUE_ID"]

        selected_indices_up_until_now = []
        for ix, r in row[col_names_without_exp_unique_id].items():
            if batch == False:
                selected_indices_up_until_now.append(r)
                selected_indices = [a for b in selected_indices_up_until_now for a in b]
            else:
                selected_indices = r

            if len(selected_indices) == 0:
                continue

            selected_classes = self.y_true[EXP_DATASET][selected_indices]
            k, class_distributions = np.unique(selected_classes, return_counts=True)

            if len(k) != len(
                self.true_distribution[EXP_DATASET][train_test_split_number]
            ):
                new_class_distributions = [0 for _ in self.classes[EXP_DATASET]]
                for key, value in zip(k, class_distributions):
                    new_class_distributions[self.classes[EXP_DATASET].index(key)] = (
                        value
                    )
                class_distributions = new_class_distributions

            class_distributions = [
                x / sum(class_distributions) for x in class_distributions
            ]

            true_counts = self.true_distribution[EXP_DATASET][train_test_split_number]
            pred_counts = class_distributions

            if metric == "manhattan":
                total_diff = 0
                for true_count, pred_count in zip(true_counts, pred_counts):
                    # distance metrics: https://stats.stackexchange.com/a/151362
                    total_diff += abs(true_count - pred_count)
                result = total_diff
            elif metric == "chebyshev":
                result = distance.chebyshev(true_counts, pred_counts)

            row[ix] = -result
        return row

    def class_distributions_manhattan_batch(
        self, row: pd.Series, EXP_DATASET: DATASET
    ) -> pd.Series:
        return self._class_distributions(
            row, metric="manhattan", batch=True, EXP_DATASET=EXP_DATASET
        )

    def unifclass_distributions_chebyshev_batch(
        self, row: pd.Series, EXP_DATASET: DATASET
    ) -> pd.Series:
        return self._class_distributions(
            row, metric="chebyshev", batch=True, EXP_DATASET=EXP_DATASET
        )

    def class_distributions_manhattan_added_up(
        self, row: pd.Series, EXP_DATASET: DATASET
    ) -> pd.Series:
        return self._class_distributions(
            row, metric="manhattan", batch=False, EXP_DATASET=EXP_DATASET
        )

    def unifclass_distributions_chebyshev_added_up(
        self, row: pd.Series, EXP_DATASET: DATASET
    ) -> pd.Series:
        return self._class_distributions(
            row, metric="chebyshev", batch=False, EXP_DATASET=EXP_DATASET
        )

    def compute_metrics(self, exp_dataset: DATASET, exp_strategy: AL_STRATEGY):
        self._per_dataset_hook(exp_dataset)

        self._compute_single_metric_jobs(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_chebyshev_batch",
            apply_to_row=self.unifclass_distributions_chebyshev_batch,
            exp_dataset=exp_dataset,
            exp_strategy=exp_strategy,
        )
        self._compute_single_metric_jobs(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_manhattan_batch",
            apply_to_row=self.class_distributions_manhattan_batch,
            exp_dataset=exp_dataset,
            exp_strategy=exp_strategy,
        )
        self._compute_single_metric_jobs(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_chebyshev_added_up",
            apply_to_row=self.unifclass_distributions_chebyshev_added_up,
            exp_dataset=exp_dataset,
            exp_strategy=exp_strategy,
        )
        self._compute_single_metric_jobs(
            existing_metric_names=["selected_indices"],
            new_metric_name="class_distributions_manhattan_added_up",
            apply_to_row=self.class_distributions_manhattan_added_up,
            exp_dataset=exp_dataset,
            exp_strategy=exp_strategy,
        )
