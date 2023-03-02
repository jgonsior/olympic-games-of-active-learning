from __future__ import annotations
import ast

from enum import unique
from enum import IntEnum
from pathlib import Path
from aenum import extend_enum
from typing import Any, Dict, Tuple
import pandas as pd
import yaml
import pandas as pd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from misc.config import (
        Config,
    )

    from resources.data_types import (
        SampleIndiceList,
        FeatureVectors,
        LabelList,
    )


@unique
class DATASET(IntEnum):
    pass


# load dataset names from yaml files
with open("resources/datasets.yaml", "r") as params_file:
    datasets_yaml_parameter_dict: Dict[str, Any] = yaml.safe_load(params_file)

for dataset_name in datasets_yaml_parameter_dict:
    extend_enum(DATASET, dataset_name)


def load_dataset(dataset: DATASET, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_file = dataset.name + ".csv"
    dataset_path: Path = config.DATASETS_PATH / dataset_file

    # load train_test_split

    train_test_split = pd.read_csv(
        config.DATASETS_PATH
        / str(dataset.name + config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX)
    )

    return pd.read_csv(dataset_path), train_test_split


def split_dataset(
    dataset: Tuple[pd.DataFrame, pd.DataFrame],
    config: Config,
) -> Tuple[
    FeatureVectors,
    LabelList,
    SampleIndiceList,
    SampleIndiceList,
    SampleIndiceList,
]:
    df, train_test_split = dataset
    X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
    Y = df["LABEL_TARGET"].to_numpy()  # type: ignore

    train_idx: SampleIndiceList = ast.literal_eval(
        train_test_split.iloc[config.EXP_TRAIN_TEST_BUCKET_SIZE]["train"]
    )
    test_idx: SampleIndiceList = ast.literal_eval(
        train_test_split.iloc[config.EXP_TRAIN_TEST_BUCKET_SIZE]["test"]
    )
    print(train_test_split.iloc[config.EXP_TRAIN_TEST_BUCKET_SIZE]["start_points"])
    initially_labeled_indices: SampleIndiceList = ast.literal_eval(
        train_test_split.iloc[config.EXP_TRAIN_TEST_BUCKET_SIZE]["start_points"]
    )[config.EXP_START_POINT]

    return X, Y, train_idx, test_idx, initially_labeled_indices
