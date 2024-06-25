from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from sklearn import datasets

from datasets.base import Base_Dataset_Loader
from misc.config import Config
from misc.logging import log_it


class OpenML_Loader(Base_Dataset_Loader):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.parameter_dict: Dict[str, Any] = yaml.safe_load(
            config.OPENML_DATASETS_YAML_CONFIG_PATH.read_text()
        )

    def load_single_dataset(
        self, dataset_name: str, dataset_raw_path: Path
    ) -> pd.DataFrame:
        log_it(
            "Dowloading {} to {}".format(
                dataset_name,
                dataset_raw_path,
            )
        )
        df = datasets.fetch_openml(
            data_id=self.parameter_dict[dataset_name]["data_id"], parser="auto"
        )["frame"]

        return df
