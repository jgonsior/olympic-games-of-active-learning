import pathlib
from typing import Any, Dict

import pandas as pd
import yaml

from misc.config import Config


class Kaggle:
    def __init__(self, config: Config) -> None:
        self.config = config
        with open("dataset_parameters.yaml", "r") as params_file:
            self.parameter_dict: Dict[str, Any] = yaml.safe_load(params_file)

    def download_datasets(self) -> None:
        for dataset_name in self.parameter_dict.keys():
            self.download_dataset(dataset_name)

    def download_dataset(self, dataset_name: str) -> None:
        datasets_raw_path = self.config.RAW_DATASETS_PATH
        datasets_cleaned_path = self.config.DATASETS_PATH

        parsing_args = self.parameter_dict[dataset_name]

        with open(datasets_raw_path, "r") as f:
            df: pd.DataFrame = pd.read_csv(f, sep=",")

        if parsing_args["drop_columns"] is not None:
            df.drop(parsing_args["drop_columns"], axis=1, inplace=True)

        for column, dtype in df.dtypes.items():  # type: ignore
            if dtype not in ["int64", "float64"]:
                if dtype.name != "category":
                    df[column] = df[column].astype("category")
                df[column] = df[column].cat.codes  # type: ignore

        label_column = parsing_args["target"]
        if label_column is not None:
            if isinstance(label_column, list):
                labels = df[label_column]
            else:
                labels = df[label_column].to_frame()
            df.drop(label_column, axis=1, inplace=True)
        else:
            labels = None

        datasets_cleaned_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            datasets_cleaned_path / parsing_args["save_name"] + "_dataset.csv",
            index=False,
        )

        if labels is not None:
            labels.to_csv(
                datasets_cleaned_path / parsing_args["save_name"] + "_label.csv",
                index=False,
            )
