from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from sklearn.preprocessing import MinMaxScaler, RobustScaler

if TYPE_CHECKING:
    from misc.config import Config

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)

from resources.data_types import SampleIndiceList

if TYPE_CHECKING:
    from misc.config import Config


class Base_Dataset_Loader(ABC):
    parameter_dict: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: Config) -> None:
        self.config: Config = config

    @abstractmethod
    def load_single_dataset(
        self, dataset_name: str, dataset_raw_path: Path
    ) -> pd.DataFrame:
        ...

    def load_datasets(self) -> None:
        datasets_raw_path = self.config.RAW_DATASETS_PATH
        datasets_raw_path.mkdir(parents=True, exist_ok=True)
        datasets_cleaned_path = self.config.DATASETS_PATH
        datasets_cleaned_path.mkdir(parents=True, exist_ok=True)

        for dataset_name in self.parameter_dict.keys():
            # if dataset_name != "arrythmia":
            #    continue
            destination_path = dataset_name + ".csv"

            dataset_raw_path = datasets_raw_path / destination_path
            dataset_clean_path = datasets_cleaned_path / destination_path

            if not dataset_raw_path.exists():
                df = self.load_single_dataset(dataset_name, dataset_raw_path)
            else:
                df = pd.read_csv(dataset_raw_path)

            if not dataset_clean_path.exists():
                df = self.preprocess_dataframe(df, dataset_name)
                splits = self.calculate_train_test_splits(dataset_name, df)

                self.save_dataset_and_splits(
                    df, splits, dataset_name, dataset_clean_path
                )

    def _map_float_to_int_categorical(self, df: pd.DataFrame, feature: str):
        feature_col = df[feature].dropna()
        unique_values = feature_col.unique()
        mapping_dict = {c: i for i, c in enumerate(unique_values)}

        return df[feature].map(mapping_dict)

    def preprocess_dataframe(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        parsing_args = self.parameter_dict[dataset_name]
        if parsing_args["drop_columns"] is not None:
            df.drop(parsing_args["drop_columns"], axis=1, inplace=True)

        # map float categoricals into int categoricals
        for feature in df.columns:
            if df[feature].dtype != float:
                continue
            feature_col = df[feature].dropna()
            diffs = np.unique(np.diff(np.sort(feature_col.unique())))
            if len(diffs) == 1 and (diffs[0] % 1) != 0:
                df[feature] = self._map_float_to_int_categorical(df, feature)

        # Special cases of float categoricals
        if "float_categorical_cols" in parsing_args:
            for feature in parsing_args["float_categorical_cols"]:
                df[feature] = self._map_float_to_int_categorical(df, feature)

        for column, dtype in df.dtypes.items():  # type: ignore
            if dtype not in ["int64", "float64"]:
                if dtype.name != "category":
                    df[column] = df[column].astype("category")
                df[column] = df[column].cat.codes  # type: ignore

        label_column = parsing_args["target"]
        assert label_column in df.columns
        df.rename(columns={label_column: "LABEL_TARGET"}, inplace=True)

        # remove rows having NaN values
        len_before = len(df)

        df.dropna(inplace=True)
        len_after = len(df)

        print(f"{len_before} - {len_after} - {dataset_name}")
        # df.fillna(value=0, inplace=True)

        X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
        Y = df["LABEL_TARGET"].to_numpy()  # type: ignore

        # check if potential id column exists
        # if so -> remove id!

        if parsing_args["id"] != None:
            df = df.drop(parsing_args["id"], axis=1)

        potential_id_columns = df.diff().fillna(0).abs().sum() / (len(df) - 1)
        potential_id_columns = potential_id_columns[potential_id_columns == 1.0].keys()

        if len(potential_id_columns) > 0:
            print(
                f"Attention! Potential Id column for dataset {dataset_name} in column {potential_id_columns}"
            )

        scaler = RobustScaler()
        X = scaler.fit_transform(X)

        # scale back to [0,1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        df = pd.DataFrame(X)
        df["LABEL_TARGET"] = Y

        return df

    """
    Creates a train_test_split.csv file, which contains for each specified split (0-9 or 0-4) the indices for the train, and the indices for the test split
    """

    def calculate_train_test_splits(
        self, dataset_name: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        # remove all classes, where we have less than 2 samples
        value_counts = df["LABEL_TARGET"].value_counts()

        for label, count in value_counts.items():
            if count <= 2:
                df = df[df.LABEL_TARGET != label]

        X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
        Y = df["LABEL_TARGET"].to_numpy()  # type: ignore
        classes = np.unique(Y)

        no_good_split_found = 0
        splits: Dict[int, Tuple[SampleIndiceList, SampleIndiceList]] = {}

        while no_good_split_found < self.config.DATASETS_AMOUNT_OF_SPLITS:
            for split_number, (train_index, test_index) in enumerate(
                StratifiedShuffleSplit(
                    n_splits=self.config.DATASETS_AMOUNT_OF_SPLITS,
                    test_size=self.config.DATASETS_TEST_SIZE_PERCENTAGE,
                ).split(X, Y)
            ):
                # quick test that all classes are present in all test_index sets
                if np.setdiff1d(np.unique(Y[test_index]), classes) == 0:  # type: ignore
                    no_good_split_found = 0
                    splits = {}
                no_good_split_found += 1
                splits[split_number] = (train_index.tolist(), test_index.tolist())  # type: ignore
        splits_df = pd.DataFrame(splits).T
        splits_df.columns = ["train", "test"]

        splits_df["start_points"] = [None for _ in range(0, len(splits_df))]

        for ix, split in splits_df.iterrows():
            start_points: List[List[int]] = []

            indices_per_label = {}

            train_copy = np.copy(split["train"])
            for label in np.unique(Y):
                indices_per_label[label] = train_copy[np.where(Y[train_copy] == label)]

            for _ in range(0, self.config.AMOUNT_OF_START_POINTS_TO_GENERATE):
                # first, pick one random sample from each class
                # then pick as much other samples as needed
                # check if we have that already, if not, we have a new start set!
                _starting_points: List[int] = []

                for label in np.unique(Y):  # type: ignore
                    # pick a random sample from that
                    index_of_start_sample_of_that_class = np.random.choice(
                        indices_per_label[label]
                    )
                    _starting_points.append(index_of_start_sample_of_that_class)

                start_points.append(_starting_points)

            splits_df.loc[ix, "start_points"] = str(start_points)

        # quick check that everything worked as intented
        """for split in range(0, 5):
            print(len(splits_df.iloc[split]["train"]) / len(X))
            print(len(splits_df.iloc[split]["test"]) / len(X))

            print(np.unique(Y[splits_df.iloc[split]["train"]], return_counts=True))
            print(np.unique(Y[splits_df.iloc[split]["test"]], return_counts=True))
        """
        return splits_df

    def save_dataset_and_splits(
        self,
        dataset_df: pd.DataFrame,
        splits_df: pd.DataFrame,
        dataset_name: str,
        dataset_clean_path: Path,
    ) -> None:
        splits_df.to_csv(
            str(dataset_clean_path).replace(".csv", "")
            + self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX,
            index=False,
        )
        dataset_df.to_csv(dataset_clean_path, index=False)
