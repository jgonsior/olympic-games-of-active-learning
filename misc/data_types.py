from typing import Dict, List, Set, Tuple
import numpy as np
from enum import IntEnum, unique
from alipy.query_strategy.query_labels import (
    QueryInstanceUncertainty,
    QueryInstanceRandom,
    QueryInstanceQUIRE,
)
from alipy.query_strategy.base import BaseIndexQuery

SampleIndiceList = List[int]
FeatureVectors = np.ndarray
LabelList = np.ndarray

# TODO move to yaml file, same as for datasets
@unique
class AL_STRATEGY(IntEnum):
    ALIPY_RANDOM = 1
    ALIPY_UNCERTAINTY_LC = 2
    ALIPY_UNCERTAINTY_ENT = 3
    ALIPY_UNCERTAINTY_MM = 4
    ALIPY_UNCERTAINTY_QUIRE = 5


strategy_mapping = {
    AL_STRATEGY.ALIPY_UNCERTAINTY_LC: (
        QueryInstanceUncertainty,
        {"measure": "least_confident"},
    ),
    AL_STRATEGY.ALIPY_UNCERTAINTY_ENT: (
        QueryInstanceUncertainty,
        {"measure": "entropy"},
    ),
    AL_STRATEGY.ALIPY_UNCERTAINTY_MM: (
        QueryInstanceUncertainty,
        {"measure": "margin"},
    ),
    AL_STRATEGY.ALIPY_UNCERTAINTY_QUIRE: (
        QueryInstanceQUIRE,
        {},
    ),
    AL_STRATEGY.ALIPY_RANDOM: (
        QueryInstanceRandom,
        {},
    ),
}

# TODO move to yaml file, same as for datasets
@unique
class SKLEARN_ML_MODELS(IntEnum):
    RF = 1
    DT = 2
    NB = 3
    SVM = 4


@unique
class AL_FRAMEWORK(IntEnum):
    ALIPY = 1
