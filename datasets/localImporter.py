from __future__ import annotations

from typing import Any, TYPE_CHECKING

from typing import Any, Dict


import pandas as pd
import yaml
from datasets.base import Base_Dataset_Loader
import shutil

if TYPE_CHECKING:
    from misc.config import Config


class Local_Importer(Base_Dataset_Loader):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if config.LOCAL_DATASETS_YAML_CONFIG_PATH.exists():
            self.parameter_dict: Dict[str, Any] = yaml.safe_load(
                config.LOCAL_DATASETS_YAML_CONFIG_PATH.read_text()
            )
        if self.parameter_dict != None:
            for k in self.parameter_dict.keys():
                self.parameter_dict[k]["id"] = None
                self.parameter_dict[k]["drop_columns"] = None
        else:
            self.parameter_dict = {}

    def load_single_dataset(
        self, dataset_name: str, dataset_raw_path: Path
    ) -> pd.DataFrame:
        shutil.copy(self.parameter_dict[dataset_name]["path"], dataset_raw_path)

        df = pd.read_csv(dataset_raw_path, sep=",")
        return df

    def calculate_train_test_splits(
        self, dataset_name: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        if self.parameter_dict[dataset_name]["train_test_splits"] == "None":
            return super().calculate_train_test_splits(dataset_name, df)
        else:
            pre_defined_splits = self.parameter_dict[dataset_name]["train_test_splits"]
            splits_df = pd.DataFrame(pre_defined_splits)
            return splits_df
