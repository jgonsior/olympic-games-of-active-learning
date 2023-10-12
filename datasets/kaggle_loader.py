import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict

import kaggle
import pandas as pd
import yaml

from datasets.base import Base_Dataset_Loader
from misc.config import Config
from misc.logging import log_it


class Kaggle_Loader(Base_Dataset_Loader):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.parameter_dict: Dict[str, Any] = yaml.safe_load(
            config.KAGGLE_DATASETS_YAML_CONFIG_PATH.read_text()
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

        kaggle.api.dataset_download_file(
            dataset=self.parameter_dict[dataset_name]["kaggle_name"],
            file_name=self.parameter_dict[dataset_name]["kaggle_file"],
            path=dataset_raw_path.parent,
        )

        potential_zip_file = (
            dataset_raw_path.parent
            / f"{self.parameter_dict[dataset_name]['kaggle_file']}.zip"
        )
        if potential_zip_file.exists():
            with zipfile.ZipFile(potential_zip_file, "r") as zip_ref:
                zip_ref.extractall(dataset_raw_path.parent)
            potential_zip_file.unlink()

        shutil.move(
            dataset_raw_path.parent / self.parameter_dict[dataset_name]["kaggle_file"],
            dataset_raw_path,
        )


        if "separator" in self.parameter_dict[dataset_name].keys():
            sep = self.parameter_dict[dataset_name]["separator"]
        else:
            sep = ","

        df = pd.read_csv(dataset_raw_path, sep=sep)
        return df
