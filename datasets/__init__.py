from __future__ import annotations

from enum import unique
from enum import IntEnum
from pathlib import Path
from aenum import extend_enum
from typing import Any, Dict, Tuple
import pandas as pd
import yaml

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from misc.config import Config


@unique
class Dataset(IntEnum):
    pass


# load dataset names from yaml files
with open("datasets/kaggle_parameters.yaml", "r") as params_file:
    datasets_yaml_parameter_dict: Dict[str, Any] = yaml.safe_load(params_file)

for dataset_name in datasets_yaml_parameter_dict:
    extend_enum(Dataset, dataset_name)


def load_dataset(dataset: Dataset, config: Config) -> pd.DataFrame:
    dataset_file = dataset.name + ".csv"
    dataset_path: Path = config.DATASETS_PATH / dataset_file
    return pd.read_csv(dataset_path)
