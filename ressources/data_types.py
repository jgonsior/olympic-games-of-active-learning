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

from framework_runners.alipy import ALIPY_AL_Experiment
from framework_runners.optimal import OPTIMAL_AL_Experiment

from optimal_query_strategies.greedy_optimal import (
    FuturePeakEvalMetric,
    GreedyHeuristic,
    Greedy_Optimal,
)

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
    OPTIMAL_BSO = 6
    OPTIMAL_TRUE = 7
    OPTIMAL_SUBSETS = 8
    OPTIMAL_GREEDY_F1 = 9
    OPTIMAL_GREEDY_F1_COS = 10
    OPTIMAL_GREEDY_F1_UNC = 11
    OPTIMAL_GREEDY_ACC = 12
    OPTIMAL_GREEDY_ACC_COS = 13
    OPTIMAL_GREEDY_ACC_UNC = 14


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
    AL_STRATEGY.OPTIMAL_GREEDY_F1: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.RANDOM,
            "future_peak_eval_metric": FuturePeakEvalMetric.F1,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_F1_COS: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.COSINE,
            "future_peak_eval_metric": FuturePeakEvalMetric.F1,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_F1_UNC: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.UNCERTAINTY,
            "future_peak_eval_metric": FuturePeakEvalMetric.F1,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_ACC: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.RANDOM,
            "future_peak_eval_metric": FuturePeakEvalMetric.ACC,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_ACC_COS: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.COSINE,
            "future_peak_eval_metric": FuturePeakEvalMetric.ACC,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_ACC_UNC: (
        Greedy_Optimal,
        {
            "heuristic": GreedyHeuristic.UNCERTAINTY,
            "future_peak_eval_metric": FuturePeakEvalMetric.ACC,
        },
    ),
    # AL_STRATEGY.OPTIMAL_BSO: (, {}),
    # AL_STRATEGY.OPTIMAL_TRUE: (, {}),
    # AL_STRATEGY.OPTIMAL_SUBSETS: (, {}),
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
