from enum import unique
from enum import IntEnum
from aenum import extend_enum
from typing import Any, Dict, Tuple
import pandas as pd
import yaml


@unique
class Dataset(IntEnum):
    pass


# load dataset names from yaml files
with open("datasets/kaggle_parameters.yaml", "r") as params_file:
    datasets_yaml_parameter_dict: Dict[str, Any] = yaml.safe_load(params_file)

for dataset_name in datasets_yaml_parameter_dict:
    extend_enum(Dataset, dataset_name)
