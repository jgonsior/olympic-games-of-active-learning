import multiprocessing
from typing import Any, Callable, Dict, List, Set, Tuple
import numpy as np
from enum import IntEnum, unique
from alipy.query_strategy.query_labels import (
    QueryInstanceUncertainty,
    QueryInstanceRandom,
    QueryInstanceQUIRE,
)
from alipy.query_strategy.base import BaseIndexQuery
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

SampleIndiceList = List[int]
FeatureVectors = np.ndarray
LabelList = np.ndarray


@unique
class AL_STRATEGY(IntEnum):
    ALIPY_RANDOM = 1
    ALIPY_UNCERTAINTY_LC = 2
    ALIPY_UNCERTAINTY_ENT = 3
    ALIPY_UNCERTAINTY_MM = 4
    ALIPY_UNCERTAINTY_QUIRE = 5


al_strategy_to_python_classes_mapping: Dict[
    AL_STRATEGY, Tuple[Callable, Dict[Any, Any]]
] = {
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


@unique
class LEARNER_MODEL(IntEnum):
    RF = 1
    DT = 2
    MNB = 3
    GNB = 4
    RBF_SVM = 5
    LINEAR_SVM = 6
    POLY_SVM = 7
    MLP = 8
    LBFGS_MLP = 9


learner_models_to_classes_mapping: Dict[
    LEARNER_MODEL, Tuple[Callable, Dict[Any, Any]]
] = {
    LEARNER_MODEL.RF: (RandomForestClassifier, {"n_jobs": multiprocessing.cpu_count()}),
    LEARNER_MODEL.DT: (DecisionTreeClassifier, {"n_jobs": multiprocessing.cpu_count()}),
    LEARNER_MODEL.MNB: (MultinomialNB, {}),
    LEARNER_MODEL.GNB: (GaussianNB, {}),
    LEARNER_MODEL.RBF_SVM: (
        SVC,
        {"kernel": "rbf", "n_jobs": multiprocessing.cpu_count()},
    ),
    LEARNER_MODEL.LINEAR_SVM: (
        SVC,
        {"kernel": "linear", "n_jobs": multiprocessing.cpu_count()},
    ),
    LEARNER_MODEL.POLY_SVM: (
        SVC,
        {"kernel": "poly", "n_jobs": multiprocessing.cpu_count()},
    ),
    LEARNER_MODEL.MLP: (
        MLPClassifier,
        {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
            "n_jobs": multiprocessing.cpu_count(),
        },
    ),  # default values
    LEARNER_MODEL.MLP: (
        MLPClassifier,
        {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "lbfgs",
            "n_jobs": multiprocessing.cpu_count(),
        },
    ),
}


@unique
class AL_FRAMEWORK(IntEnum):
    ALIPY = 1
