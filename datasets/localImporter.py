from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, List

from typing import Any, Dict

from flask import render_template
import glob

import pandas as pd
from datasets.base import Base_Dataset_Loader


if TYPE_CHECKING:
    from misc.config import Config


class Local_Importer(Base_Dataset_Loader):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.local_datasets_path = config.DATASETS_PATH.parent / "local_datasets"

        for local_csv in glob.glob(str(self.local_datasets_path) + "/*.csv"):
            self.parameter_dict[local_csv.replace(".csv", "")] = {
                "target": "Color",
                "drop_columns": None,
                id: None,
            }

    def load_single_dataset(
        self, dataset_name: str, dataset_raw_path: Path
    ) -> pd.DataFrame:
        # copy over to dataset_raw_path
        shutil.copy(self.local_datasets_path / f"{dataset_name}.csv", dataset_raw_path)
        df = pd.read_csv(dataset_raw_path, sep=",")
        return df
