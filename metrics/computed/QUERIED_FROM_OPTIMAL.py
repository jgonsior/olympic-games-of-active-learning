from __future__ import annotations
import ast
from pathlib import Path
from statistics import harmonic_mean
import numpy as np
import pandas as pd
from datasets import DATASET
from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    pass


class QUERIED_FROM_OPTIMAL(Base_Computed_Metric):
    optimal_samples_order_variability: np.ndarray
    optimal_samples_order_wrongness: np.ndarray
    optimal_samples_order_easy_hard_ambiguous: np.ndarray

    optimal_samples_order_acc_diff_addition: np.ndarray
    optimal_samples_included_in_optimal_strategy: np.ndarray

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        optimal_samples_path = Path(
            f"{self.config.OUTPUT_PATH}/_optimal_samples/{EXP_DATASET.name}.npz"
        )

        # 4 ways to calculate optimum:
        # per sample: sum up the +acc changes -> this is the "order weight"?
        # variability among +-acc per batch

        # alternatively "datamaps"
        # calculate, how often the sample has beeng predicted correctly at all times
        # calculate the variability among the predictions
        # optimal being part of OPTIMAL_GREEDY query
        if not optimal_samples_path.exists():
            print("Computing ", optimal_samples_path)
            y_train_true: Dict[int, np.ndarray] = {}
            y_test_true: Dict[int, np.ndarray] = {}

            y = pd.read_csv(
                f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}.csv",
                usecols=["LABEL_TARGET"],
            )["LABEL_TARGET"].to_numpy()

            wrongly_classified_counter = {ix: [] for ix in range(0, len(y))}
            acc_diff_per_indice_counts = {ix: [] for ix in range(0, len(y))}
            self.optimal_samples_included_in_optimal_strategy = np.zeros_like(
                y, dtype=np.float16
            )

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

            # @TODO multiprocessing
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

                selected_indices_path = Path(
                    self.config.OUTPUT_PATH
                    / EXP_STRATEGY.name
                    / EXP_DATASET.name
                    / "selected_indices.csv.xz"
                )
                selected_indices = pd.read_csv(selected_indices_path, header=0).melt(
                    id_vars="EXP_UNIQUE_ID",
                    var_name="al_cycle",
                    value_name="selected_indices",
                )

                selected_indices["selected_indices"] = selected_indices[
                    "selected_indices"
                ].fillna("[]")

                selected_indices["selected_indices"] = selected_indices[
                    "selected_indices"
                ].apply(ast.literal_eval)

                if "OPTIMAL_" in EXP_STRATEGY.name:
                    # ignore start set
                    for row in selected_indices.loc[
                        selected_indices["al_cycle"] != "0"
                    ]["selected_indices"].to_list():
                        for si in row:
                            self.optimal_samples_included_in_optimal_strategy[si] += 1

                acc_path = Path(
                    self.config.OUTPUT_PATH
                    / EXP_STRATEGY.name
                    / EXP_DATASET.name
                    / "accuracy.csv.xz"
                )

                accs = pd.read_csv(acc_path, header=0)
                for row_ix in range(len(accs.columns) - 2, 0, -1):
                    accs[str(row_ix)] = accs[str(row_ix)] - accs[str(row_ix - 1)]
                del accs["0"]

                accs = accs.melt(
                    id_vars="EXP_UNIQUE_ID", var_name="al_cycle", value_name="accs"
                )

                accs = accs.merge(
                    selected_indices, on=["EXP_UNIQUE_ID", "al_cycle"], how="inner"
                )

                # we list per sample, how it changed the accuracy
                for _, row in accs.iterrows():
                    acc_diff = row["accs"]
                    selected_indices = row["selected_indices"]

                    for ix in selected_indices:
                        acc_diff_per_indice_counts[ix].append(acc_diff)

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

                    if row["y"] is np.nan:
                        continue

                    y_pred_single = np.array(ast.literal_eval(row["y"]))

                    if train_or_test.endswith("_train"):
                        y_true_single = y_train_true[EXP_TRAIN_TEST_BUCKET_SIZE]
                        global_indices = train_splits[EXP_TRAIN_TEST_BUCKET_SIZE]
                    else:
                        y_true_single = y_test_true[EXP_TRAIN_TEST_BUCKET_SIZE]
                        global_indices = test_splits[EXP_TRAIN_TEST_BUCKET_SIZE]

                    for local_ix, global_ix in enumerate(global_indices):
                        if y_pred_single[local_ix] == y_true_single[local_ix]:
                            wrongly_classified_counter[global_ix].append(0)
                        else:
                            wrongly_classified_counter[global_ix].append(1)
            self.optimal_samples_order_wrongness = np.zeros_like(y, dtype=np.float16)
            self.optimal_samples_order_variability = np.zeros_like(y, dtype=np.float16)
            self.optimal_samples_order_easy_hard_ambiguous = np.zeros_like(
                y, dtype=np.float16
            )

            self.optimal_samples_order_acc_diff_addition = np.zeros_like(
                y, dtype=np.float16
            )

            for global_ix, values in wrongly_classified_counter.items():
                self.optimal_samples_order_wrongness[global_ix] = np.sum(values) / len(
                    values
                )
                self.optimal_samples_order_variability[global_ix] = np.std(values)
                self.optimal_samples_order_easy_hard_ambiguous[
                    global_ix
                ] = harmonic_mean(
                    [
                        self.optimal_samples_order_wrongness[global_ix],
                        self.optimal_samples_order_variability[global_ix],
                    ]
                )

                optimal_samples_path.parent.mkdir(exist_ok=True)
            print(self.optimal_samples_order_wrongness)
            for global_ix, values in acc_diff_per_indice_counts.items():
                self.optimal_samples_order_acc_diff_addition[global_ix] = np.sum(values)

            exit(-1)
            np.savez_compressed(
                optimal_samples_path,
                optimal_samples_order_wrongness=self.optimal_samples_order_wrongness,
                optimal_samples_order_variability=self.optimal_samples_order_variability,
                optimal_samples_order_easy_hard_ambiguous=self.optimal_samples_order_easy_hard_ambiguous,
                optimal_samples_order_acc_diff_addition=self.optimal_samples_order_acc_diff_addition,
                optimal_samples_included_in_optimal_strategy=self.optimal_samples_included_in_optimal_strategy,
            )
        else:
            print("Loading ", optimal_samples_path)
            self.optimal_samples_order_acc_diff_addition = np.load(
                optimal_samples_path
            )["optimal_samples_order_acc_diff_addition"]
            self.optimal_samples_order_wrongness = np.load(optimal_samples_path)[
                "optimal_samples_order_wrongness"
            ]
            self.optimal_samples_order_variability = np.load(optimal_samples_path)[
                "optimal_samples_order_variability"
            ]
            self.optimal_samples_order_easy_hard_ambiguous = np.load(
                optimal_samples_path
            )["optimal_samples_order_easy_hard_ambiguous"]
            self.optimal_samples_included_in_optimal_strategy = np.load(
                optimal_samples_path
            )["optimal_samples_included_in_optimal_strategy"]

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        del df["0"]
        return self._convert_selected_indices_to_ast(df)

    def _optimal_samples_order_wrongness(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return -np.sum(self.optimal_samples_order_wrongness[row["selected_indices"]])

    def _optimal_samples_order_variability(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return -np.sum(self.optimal_samples_order_variability[row["selected_indices"]])

    def _optimal_samples_order_easy_hard_ambiguous(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return -np.sum(
            self.optimal_samples_order_easy_hard_ambiguous[row["selected_indices"]]
        )

    def _optimal_samples_order_acc_diff_addition(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return -np.sum(
            self.optimal_samples_order_acc_diff_addition[row["selected_indices"]]
        )

    def _optimal_samples_included_in_optimal_strategy(
        self,
        row: pd.Series,
    ) -> pd.Series:
        return np.sum(
            self.optimal_samples_included_in_optimal_strategy[row["selected_indices"]]
        )

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="optimal_samples_order_wrongness",
            apply_to_row=self._optimal_samples_order_wrongness,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="optimal_samples_order_variability",
            apply_to_row=self._optimal_samples_order_variability,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="optimal_samples_order_easy_hard_ambiguous",
            apply_to_row=self._optimal_samples_order_easy_hard_ambiguous,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="optimal_samples_order_acc_diff_addition",
            apply_to_row=self._optimal_samples_order_acc_diff_addition,
        )

        self._take_single_metric_and_compute_new_one(
            existing_metric_names=["selected_indices"],
            new_metric_name="optimal_samples_included_in_optimal_strategy",
            apply_to_row=self._optimal_samples_included_in_optimal_strategy,
        )
