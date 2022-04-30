from __future__ import annotations

from enum import unique
from enum import IntEnum
from pathlib import Path
from aenum import extend_enum
from typing import Any, Dict, Tuple, Union, cast
import pandas as pd
import yaml
import math
from typing import List
import pandas as pd
from misc.logging import log_it
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from misc.config import (
        Config,
    )

from ressources.data_types import (
    SampleIndiceList,
    AL_STRATEGY,
    FeatureVectors,
    LabelList,
    AL_FRAMEWORK,
)


@unique
class DATASET(IntEnum):
    pass


# load dataset names from yaml files
with open("ressources/datasets.yaml", "r") as params_file:
    datasets_yaml_parameter_dict: Dict[str, Any] = yaml.safe_load(params_file)

for dataset_name in datasets_yaml_parameter_dict:
    extend_enum(DATASET, dataset_name)


def load_dataset(dataset: DATASET, config: Config) -> pd.DataFrame:
    dataset_file = dataset.name + ".csv"
    dataset_path: Path = config.DATASETS_PATH / dataset_file
    return pd.read_csv(dataset_path)


def split_dataset(
    df: pd.DataFrame,
    config: Config,
    al_framework: AL_FRAMEWORK = AL_FRAMEWORK.ALIPY,
) -> Tuple[
    FeatureVectors,
    LabelList,
    SampleIndiceList,
    SampleIndiceList,
    SampleIndiceList,
    SampleIndiceList,
]:
    X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
    Y = df["LABEL_TARGET"].to_numpy()  # type: ignore

    shuffling = np.random.permutation(len(Y))
    X = X[shuffling]
    Y = Y[shuffling]

    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    # scale back to [0,1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # fancy ALiPy train/test split
    test_ratio = cast(float, config.EXP_TRAIN_TEST_SPLIT)
    indices: SampleIndiceList = [i for i in range(0, len(Y))]
    train_idx: SampleIndiceList = indices[: math.floor(len(Y) * (1 - test_ratio))]
    test_idx: SampleIndiceList = indices[math.floor(len(Y) * (1 - test_ratio)) :]
    unlabel_idx: SampleIndiceList = train_idx.copy()
    label_idx: SampleIndiceList = []

    for label in np.unique(Y):  # type: ignore
        if label not in Y[train_idx]:  # type: ignore
            print(np.where(Y[test_idx] == label))
        init_labeled_index = np.where(Y[train_idx] == label)[0][0]  # type: ignore
        label_idx.append(init_labeled_index)
        unlabel_idx.remove(init_labeled_index)

    return X, Y, train_idx, test_idx, label_idx, unlabel_idx
