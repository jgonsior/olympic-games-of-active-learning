from __future__ import annotations
import multiprocessing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
import numpy as np
from enum import IntEnum, unique
from alipy.query_strategy import (
    QueryInstanceUncertainty,
    QueryInstanceRandom,
    QueryInstanceQUIRE,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

from framework_runners.alipy_runner import ALIPY_AL_Experiment
from framework_runners.optimal_runner import OPTIMAL_AL_Experiment
from optimal_query_strategies.BSO_optimal import Beeam_Search_Optimal

from optimal_query_strategies.greedy_optimal import (
    FuturePeakEvalMetric,
    Greedy_Optimal,
)
from optimal_query_strategies.true_optimal import True_Optimal

SampleIndiceList = List[int]
FeatureVectors = np.ndarray
LabelList = np.ndarray


@unique
class AL_STRATEGY(IntEnum):
    ALIPY_RANDOM = 1
    ALIPY_UNCERTAINTY = 2
    ALIPY_GRAPH_DENSITY = 3
    ALIPY_CORESET_GREEDY = 4
    ALIPY_QUIRE = 5
    OPTIMAL_BSO = 6
    OPTIMAL_TRUE = 7
    OPTIMAL_GREEDY = 9


al_strategy_to_python_classes_mapping: Dict[AL_STRATEGY, Callable] = {
    AL_STRATEGY.ALIPY_UNCERTAINTY: QueryInstanceUncertainty,
    AL_STRATEGY.ALIPY_QUIRE: QueryInstanceQUIRE,
    AL_STRATEGY.ALIPY_RANDOM: QueryInstanceRandom,
    AL_STRATEGY.OPTIMAL_GREEDY: Greedy_Optimal,
    AL_STRATEGY.OPTIMAL_BSO: Beeam_Search_Optimal,
    AL_STRATEGY.OPTIMAL_TRUE: True_Optimal,
    # AL_STRATEGY.OPTIMAL_SUBSETS: (, {}),
}


# TODO parameter wie für AL strats in exp_config.yaml
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
    LEARNER_MODEL.DT: (DecisionTreeClassifier, {}),
    LEARNER_MODEL.MNB: (MultinomialNB, {}),
    LEARNER_MODEL.GNB: (GaussianNB, {}),
    LEARNER_MODEL.RBF_SVM: (
        SVC,
        {
            "kernel": "rbf",
            "probability": True,
        },
    ),
    LEARNER_MODEL.LINEAR_SVM: (
        SVC,
        {
            "kernel": "linear",
            "probability": True,
        },
    ),
    LEARNER_MODEL.POLY_SVM: (
        SVC,
        {
            "kernel": "poly",
            "probability": True,
        },
    ),
    LEARNER_MODEL.MLP: (
        MLPClassifier,
        {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "adam",
        },
    ),  # default values
    LEARNER_MODEL.LBFGS_MLP: (
        MLPClassifier,
        {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "solver": "lbfgs",
        },
    ),
}


@unique
class AL_FRAMEWORK(IntEnum):
    ALIPY = 1
    OPTIMAL = 2


AL_framework_to_classes_mapping: Dict[AL_FRAMEWORK, Tuple[Callable, Dict[Any, Any]]] = {
    AL_FRAMEWORK.ALIPY: (ALIPY_AL_Experiment, {}),
    AL_FRAMEWORK.OPTIMAL: (OPTIMAL_AL_Experiment, {}),
}
