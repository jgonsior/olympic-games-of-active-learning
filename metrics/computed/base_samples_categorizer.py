from __future__ import annotations
from abc import ABC
import math
from pathlib import Path
from typing import Callable, Iterable, List, TYPE_CHECKING, Tuple
import ast
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier

from datasets import DATASET
import ast
from pathlib import Path
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
    calculates how often the predicted class changes over the course of an AL cycls by calculating the variance of the predicted classes over the course of the AL cycles
    """

    def calculate_samples_categorization(self, dataset: DATASET) -> np.ndarray:
        _, Y_true = self._load_dataset(dataset)
        samples_categorization = np.zeros_like(Y_true, dtype=np.float64)
        for Y_preds in self._get_Y_preds_iterator(dataset):
            a = Y_preds.to_numpy()

            if len(a) == 0:
                continue

            a = np.apply_along_axis(lambda x: [x[0], x[1], x[2]], axis=1, arr=a)
            a = np.var(a, axis=1)
            a = np.sum(a, axis=0)
            samples_categorization += a
        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 1000
        )
        return samples_categorization


# x
class CLOSENESS_TO_DECISION_BOUNDARY(Base_Samples_Categorizer):
    """
    use SVM to calculate exact decision boundaries -> calculate, how far away from the next decision boundary a sample is
    """

    def calculate_samples_categorization(
        self,
        dataset: DATASET,
    ) -> np.ndarray:
        from resources.data_types import (
            LEARNER_MODEL,
            learner_models_to_classes_mapping,
        )

        X, Y_true = self._load_dataset(dataset)

        learner_params = learner_models_to_classes_mapping[LEARNER_MODEL.RBF_SVM]
        learner = learner_params[0](**learner_params[1], decision_function_shape="ovr")

        learner.fit(X, Y_true)
        decision_boundary_distances = np.abs(learner.decision_function(X))

        if len(np.shape(decision_boundary_distances)) > 1:
            min_distances = np.min(decision_boundary_distances, axis=1)
        else:
            min_distances = decision_boundary_distances

        samples_categorization = min_distances

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 1000
        )

        return samples_categorization


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


# x
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
            n_neighbors=k, return_distance=False
        )

        # calculate indices to classes using Y_true
        nearest_neighbors_classes = np.apply_along_axis(
            lambda x: Y_true[x], 1, nearest_neighbors_of_each_point
        )

        nearest_variance = np.var(nearest_neighbors_classes, axis=1)
        samples_categorization = nearest_variance

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


# x
class CLOSENESS_TO_CLUSTER_CENTER(Base_Samples_Categorizer):
    """
    first, cluster dataset
    second, calculate distance of point to cluster center
    """

    def calculate_samples_categorization(
        self,
        dataset: DATASET,
    ) -> np.ndarray:
        X, Y_true = self._load_dataset(dataset)

        k = math.floor(math.sqrt(np.shape(Y_true)[0]) / 2)

        print(f"{dataset.name} k= {k}")

        clusterer = MiniBatchKMeans(
            n_clusters=k,
        )
        distances_to_clusters_centers = np.min(clusterer.fit_transform(X), axis=1)
        samples_categorization = distances_to_clusters_centers

        # normalize samples_categorization
        samples_categorization = (
            samples_categorization / np.sum(samples_categorization) * 100 * k
        )

        return samples_categorization


class IMPROVES_ACCURACY_BY(Base_Samples_Categorizer):
    """
    count how often this sample improves the accuracy, if it was part of a batch
    """

    ...
