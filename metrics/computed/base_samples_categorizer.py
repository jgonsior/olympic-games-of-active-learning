from __future__ import annotations
from abc import ABC
import math
from pathlib import Path
from typing import Callable, Iterable, List, TYPE_CHECKING, Tuple
from joblib import Parallel, delayed, parallel_backend
import ast
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale

from datasets import DATASET
import ast
from pathlib import Path
from statistics import harmonic_mean
import numpy as np
import pandas as pd
from datasets import DATASET

from typing import Dict, TYPE_CHECKING


if TYPE_CHECKING:
    from misc.config import Config
    from resources.data_types import FeatureVectors, LabelList


"""
operates on a per dataset basis
categorizes each sample of the dataset into different categories
"""


class Base_Samples_Categorizer(ABC):
    metrics: List[str]
    done_workload_df: pd.DataFrame
    _train_test_splits: Dict[DATASET, pd.DataFrame] = {}

    def __init__(self, config: Config) -> None:
        self.done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
        self.config = config

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        ...

    def categorize_samples(self, dataset: DATASET) -> None:
        samples_categorization_path = Path(
            f"{self.config.OUTPUT_PATH}/_{self.__class__.__name__}/{dataset.name}.npz"
        )
        samples_categorization_path.parent.mkdir(parents=True, exist_ok=True)

        samples_categorization = self.calculate_samples_categorization(dataset)

        np.savez_compressed(
            samples_categorization_path, samples_categorization=samples_categorization
        )

    def _load_dataset(self, dataset: DATASET) -> Tuple[FeatureVectors, LabelList]:
        X: FeatureVectors = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{dataset.name}.csv"
        ).to_numpy()

        Y_true: LabelList = pd.read_csv(
            f"{self.config.DATASETS_PATH}/{dataset.name}.csv",
            usecols=["LABEL_TARGET"],
        )["LABEL_TARGET"].to_numpy()

        return X, Y_true

    def _get_distance_matrix(self, dataset: DATASET) -> np.ndarray:
        return pd.read_csv(
            f"{self.config.DATASETS_PATH}/{dataset.name}{self.config.DATASETS_DISTANCES_APPENDIX}",
        ).to_numpy()

    def _convert_selected_indices_to_ast(self, df: pd.DataFrame) -> pd.DataFrame:
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")

        df = df.fillna("")
        df["selected_indices"] = df[column_names_which_are_al_cycles].apply(
            lambda x: ast.literal_eval(
                ("[" + ",".join(x).replace("[", "").replace("]", "") + "]").replace(
                    ",,", ""
                )
            ),
            axis=1,
        )
        for c in column_names_which_are_al_cycles:
            del df[c]

        return df

    def _convert_(self, df: pd.DataFrame) -> pd.DataFrame:
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")

        df = df.fillna("")
        df["selected_indices"] = df[column_names_which_are_al_cycles].apply(
            lambda x: ast.literal_eval(
                ("[" + ",".join(x).replace("[", "").replace("]", "") + "]").replace(
                    ",,", ""
                )
            ),
            axis=1,
        )
        for c in column_names_which_are_al_cycles:
            del df[c]

        return df

    def _get_train_test_splits(self, dataset) -> pd.DataFrame:
        if dataset not in self._train_test_splits.keys():
            self._train_test_splits[dataset] = (
                pd.read_csv(
                    f"{self.config.DATASETS_PATH}/{dataset.name}{self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}",
                    usecols=["train", "test"],
                )
                .applymap(lambda x: ast.literal_eval(x.replace(",,", "")))
                .reset_index(level=0)
                .rename(columns={"index": "EXP_TRAIN_TEST_BUCKET_SIZE"})
            )
        return self._train_test_splits[dataset]

    def _merge_indices(
        self, row: pd.Series, cols_with_indice_lists: List[int]
    ) -> List[np.ndarray]:
        result = np.empty(
            (len(cols_with_indice_lists), len(row["train"]) + len(row["test"])),
            dtype=np.int8,
        )
        result.fill(np.nan)

        for c in cols_with_indice_lists:
            if row[f"{c}_train"] == []:
                continue
            result[c, row["train"]] = row[f"{c}_train"]
            result[c, row["test"]] = row[f"{c}_test"]

        return [r for r in result]

    def _get_Y_preds_iterator(self, dataset: DATASET) -> Iterable[pd.DataFrame]:
        _train_test_splits = self._get_train_test_splits(dataset)
        for strat in self.config.EXP_GRID_STRATEGY:
            y_pred_train_path = Path(
                f"{self.config.OUTPUT_PATH}/{strat.name}/{dataset.name}/y_pred_train.csv.xz"
            )
            if not y_pred_train_path.exists():
                continue
            Y_pred_train = pd.read_csv(y_pred_train_path)
            cols_with_indice_lists = Y_pred_train.columns.difference(["EXP_UNIQUE_ID"])

            Y_pred_train[cols_with_indice_lists] = (
                Y_pred_train[cols_with_indice_lists]
                .fillna("[]")
                .applymap(lambda x: ast.literal_eval(x))
            )
            Y_pred_test = pd.read_csv(
                f"{self.config.OUTPUT_PATH}/{strat.name}/{dataset.name}/y_pred_test.csv.xz",
            )
            Y_pred_test[cols_with_indice_lists] = (
                Y_pred_test[cols_with_indice_lists]
                .fillna("[]")
                .applymap(lambda x: ast.literal_eval(x))
            )

            # get train_test_splits based on EXP_UNIQUE_ID
            exp_train_test_buckets = self.done_workload_df[
                self.done_workload_df["EXP_UNIQUE_ID"].isin(Y_pred_train.EXP_UNIQUE_ID)
            ]  # ["EXP_TRAIN_TEST_BUCKET_SIZE"].to_frame()

            Y_pred = Y_pred_train.merge(
                Y_pred_test,
                how="inner",
                on="EXP_UNIQUE_ID",
                suffixes=["_train", "_test"],
            )

            exp_train_test_buckets = exp_train_test_buckets.merge(
                _train_test_splits, how="left", on="EXP_TRAIN_TEST_BUCKET_SIZE"
            )[["EXP_UNIQUE_ID", "train", "test"]]
            Y_pred = Y_pred.merge(
                exp_train_test_buckets, how="inner", on="EXP_UNIQUE_ID"
            )

            exp_unique_ids = Y_pred["EXP_UNIQUE_ID"]
            Y_pred = Y_pred.apply(
                lambda x: self._merge_indices(
                    x, [int(c) for c in cols_with_indice_lists]
                ),
                axis=1,
                result_type="expand",
            )

            Y_pred.set_index(exp_unique_ids, inplace=True)

            yield Y_pred

    def _closeness_to_k_nearest(
        self,
        dataset: DATASET,
        mask_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)

        distance_matrix = self._get_distance_matrix(dataset)

        k = math.floor(math.sqrt(np.shape(distance_matrix)[0]) / 2)
        print(f"{dataset.name} k= {k}")

        samples_categorization = np.zeros_like(Y_true, dtype=np.float16)

        for Y_class in np.unique(Y_true):
            samples_of_this_class_mask = np.where(mask_func(Y_true, Y_class))[0]

            neigh = KNeighborsClassifier(
                n_neighbors=k,
                metric="precomputed",
            )
            neigh.fit(
                distance_matrix[samples_of_this_class_mask][
                    :, samples_of_this_class_mask
                ],
                Y_true[samples_of_this_class_mask],
            )

            nearest_neighbors_of_same_class_distances = neigh.kneighbors(
                n_neighbors=k, return_distance=True
            )[0]

            avg_distance_to_same_class_neighbors = np.average(
                nearest_neighbors_of_same_class_distances, axis=1
            )
            samples_categorization[
                samples_of_this_class_mask
            ] = avg_distance_to_same_class_neighbors

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * k
        )
        return samples_categorization


# x
class COUNT_WRONG_CLASSIFICATIONS(Base_Samples_Categorizer):
    """
    is often classified wrongly
    """

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)
        samples_categorization = np.zeros_like(Y_true)
        for Y_preds in self._get_Y_preds_iterator(dataset):
            for exp_unique_id, r in Y_preds.iterrows():
                for al_cycle_iteration, Y_pred in enumerate(r):
                    # calculate how often Y_pred and Y_true are not equal
                    samples_categorization[np.where(Y_pred != Y_true)] += 1
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 1000
        )
        return samples_categorization


# x
class SWITCHES_CLASS_OFTEN(Base_Samples_Categorizer):
    """
    calculates how often the predicted class changes often over the cours of an AL cycle
    """

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)
        samples_categorization = np.zeros_like(Y_true)
        for Y_preds in self._get_Y_preds_iterator(dataset):
            for exp_unique_id, r in Y_preds.iterrows():
                for Y_pred_previous, Y_pred_current in zip(r[0:-1], r[1:]):
                    samples_categorization[
                        np.where(Y_pred_current != Y_pred_previous)
                    ] += 1

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 1000
        )
        return samples_categorization


class CLOSENESS_TO_DECISION_BOUNDARY(Base_Samples_Categorizer):
    """
    use SVM to calculate exact decision boundaries -> calculate, how far away from the next decision boundary a sample is
    """

    ...


# x
class REGION_DENSITY(Base_Samples_Categorizer):
    """
    use kNN or so to calculate, what the average distance of a sample to its k next neighbors is
    """

    def calculate_samples_categorization(
        self,
        dataset: DATASET,
    ) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)

        distance_matrix = self._get_distance_matrix(dataset)

        k = math.floor(math.sqrt(np.shape(distance_matrix)[0]) / 2)
        print(f"{dataset.name} k= {k}")

        neigh = KNeighborsClassifier(
            n_neighbors=k,
            metric="precomputed",
        )
        neigh.fit(
            distance_matrix,
            Y_true,
        )

        nearest_neighbors_of_each_point = neigh.kneighbors(
            n_neighbors=k, return_distance=True
        )[0]

        avg_distance_to_neighbors = np.average(nearest_neighbors_of_each_point, axis=1)
        samples_categorization = avg_distance_to_neighbors

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 100 * k
        )

        return samples_categorization


class MELTING_POT_REGION(Base_Samples_Categorizer):
    """
    counts how many other classes are present among the k=5 nearest neighbors
    """

    def calculate_samples_categorization(
        self,
        dataset: DATASET,
    ) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)

        distance_matrix = self._get_distance_matrix(dataset)

        k = math.floor(math.sqrt(np.shape(distance_matrix)[0]) / 2)
        print(f"{dataset.name} k= {k}")

        neigh = KNeighborsClassifier(
            n_neighbors=k,
            metric="precomputed",
        )
        neigh.fit(
            distance_matrix,
            Y_true,
        )

        nearest_neighbors_of_each_point = neigh.kneighbors(
            n_neighbors=k, return_distance=True
        )[0]

        avg_distance_to_neighbors = np.average(nearest_neighbors_of_each_point, axis=1)
        samples_categorization = avg_distance_to_neighbors

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 100 * k
        )

        return samples_categorization


class INCLUDED_IN_OPTIMAL_STRATEGY(Base_Samples_Categorizer):
    """
    counts how often a sample is included in an optimal strategy
    """

    ...


# x
class CLOSENESS_TO_SAMPLES_OF_SAME_CLASS(Base_Samples_Categorizer):
    """
    first, cluster dataset
    second, calculate distance of point to cluster border
    """

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        return self._closeness_to_k_nearest(dataset, mask_func=lambda a, b: a == b)


# x
class CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS(Base_Samples_Categorizer):
    """
    first, cluster dataset
    second, calculate distance of point to cluster border
    """

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        return self._closeness_to_k_nearest(dataset, mask_func=lambda a, b: a != b)


class CLOSENES_TO_CLUSTER_BORDER(Base_Samples_Categorizer):
    """
    first, cluster dataset
    second, calculate distance of point to cluster border
    """

    ...


class IMPROVES_ACCURACY_BY(Base_Samples_Categorizer):
    """
    count how often this sample improves the accuracy, if it was part of a batch
    """

    ...


class EASY_TO_CLASSIFY_CATEGORIZER(Base_Samples_Categorizer):
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
