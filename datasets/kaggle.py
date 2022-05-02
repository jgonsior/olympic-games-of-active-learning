from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml
from misc.config import Config
import kaggle
from misc.logging import log_it


class Kaggle:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.parameter_dict: Dict[str, Any] = yaml.safe_load(
            config.KAGGLE_DATASETS_PATH.read_text()
        )

    def download_all_datasets(self) -> None:
        for dataset_name in self.parameter_dict.keys():
            self.download_single_dataset(dataset_name)
            self.preprocess_dataset(dataset_name)

    def download_single_dataset(self, dataset_name: str) -> None:
        destination_path: Path = (
            self.config.RAW_DATASETS_PATH
            / self.parameter_dict[dataset_name]["kaggle_file"]
        )

        if not destination_path.exists():
            log_it(
                "Dowloading {} to {}/{}".format(
                    dataset_name,
                    destination_path,
                    self.parameter_dict[dataset_name]["kaggle_file"],
                )
            )
            kaggle.api.dataset_download_file(
                dataset=self.parameter_dict[dataset_name]["kaggle_name"],
                file_name=self.parameter_dict[dataset_name]["kaggle_file"],
                path=destination_path.parent,
            )

    def preprocess_dataset(self, dataset_name: str) -> None:
        datasets_raw_path = self.config.RAW_DATASETS_PATH
        datasets_cleaned_path = self.config.DATASETS_PATH

        datasets_raw_path.mkdir(parents=True, exist_ok=True)
        destination_path = dataset_name + ".csv"
        datasets_cleaned_path = datasets_cleaned_path / destination_path

        if not datasets_cleaned_path.exists():

            parsing_args = self.parameter_dict[dataset_name]

            with open(
                datasets_raw_path / self.parameter_dict[dataset_name]["kaggle_file"],
                "r",
            ) as f:
                df: pd.DataFrame = pd.read_csv(f, sep=",")

            if parsing_args["drop_columns"] is not None:
                df.drop(parsing_args["drop_columns"], axis=1, inplace=True)

            for column, dtype in df.dtypes.items():  # type: ignore
                if dtype not in ["int64", "float64"]:
                    if dtype.name != "category":
                        df[column] = df[column].astype("category")
                    df[column] = df[column].cat.codes  # type: ignore

            label_column = parsing_args["target"]
            df.rename(columns={label_column: "LABEL_TARGET"}, inplace=True)

            df.to_csv()(
                datasets_cleaned_path,
                index=False,
            )

            log_it(f"Done Preprocessing {dataset_name} to {datasets_cleaned_path}")
