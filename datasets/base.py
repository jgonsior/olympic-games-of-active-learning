from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from misc.config import Config

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
)

from misc.logging import log_it
from resources.data_types import SampleIndiceList

if TYPE_CHECKING:
    from misc.config import Config


class Base_Dataset_Loader(ABC):
    parameter_dict: Dict[str, Dict[str, Any]] = {}

    def __init__(self, config: Config) -> None:
        super().__init__()
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
            destination_path = dataset_name + ".csv"

            dataset_raw_path = datasets_raw_path / destination_path
            dataset_clean_path = datasets_cleaned_path / destination_path

            if not dataset_raw_path.exists():
                df = self.load_single_dataset(dataset_name, dataset_raw_path)
            else:
                df = pd.read_csv(dataset_raw_path)

            if not dataset_clean_path.exists():
                df = self.preprocess_dataframe(df, dataset_name)
                splits = self.calculate_train_test_splits(df)

                self.save_dataset_and_splits(
                    df, splits, dataset_name, dataset_clean_path
                )

    def preprocess_dataframe(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        parsing_args = self.parameter_dict[dataset_name]
        if parsing_args["drop_columns"] is not None:
            df.drop(parsing_args["drop_columns"], axis=1, inplace=True)

        for column, dtype in df.dtypes.items():  # type: ignore
            if dtype not in ["int64", "float64"]:
                if dtype.name != "category":
                    df[column] = df[column].astype("category")
                df[column] = df[column].cat.codes  # type: ignore

        label_column = parsing_args["target"]
        df.rename(columns={label_column: "LABEL_TARGET"}, inplace=True)

        # remove rows having NaN values
        df.dropna(inplace=True)

        return df

    """
    Creates a train_test_split.csv file, which contains for each specified split (0-9 or 0-4) the indices for the train, and the indices for the test split
    """

    def calculate_train_test_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        # remove all classes, where we have less than 2 samples
        value_counts = df["LABEL_TARGET"].value_counts()

        for label, count in value_counts.iteritems():
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

        # quick check that everything wored as intented
        for split in range(0, 5):
            print(len(splits_df.iloc[split]["train"]) / len(X))
            print(len(splits_df.iloc[split]["test"]) / len(X))

            print(np.unique(Y[splits_df.iloc[split]["train"]], return_counts=True))
            print(np.unique(Y[splits_df.iloc[split]["test"]], return_counts=True))

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
            + self.config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX
            + ".csv",
            index=False,
        )
        dataset_df.to_csv(dataset_clean_path, index=False)
