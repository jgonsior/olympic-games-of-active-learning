from __future__ import annotations
from distutils.command.config import config
import multiprocessing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
import numpy as np
from enum import IntEnum, unique
from alipy.query_strategy import (
    QueryInstanceUncertainty,
    QueryInstanceRandom,
    QueryInstanceQUIRE,
    QueryExpectedErrorReduction,
    QueryInstanceBMDR,
    QueryInstanceCoresetGreedy,
    QueryInstanceDensityWeighted,
    QueryInstanceGraphDensity,
    QueryInstanceLAL,
    QueryInstanceQBC,
    QueryInstanceSPAL,
)
from libact.query_strategies import (
    UncertaintySampling,
    QueryByCommittee,
    DWUS,
    QUIRE,
)
from libact.models import LogisticRegression, SVM
from playground.sampling_methods.bandit_discrete import BanditDiscreteSampler  # wrapper
from playground.sampling_methods.simulate_batch import SimulateBatchSampler  # wrapper
from playground.sampling_methods.graph_density import GraphDensitySampler
from playground.sampling_methods.hierarchical_clustering_AL import HierarchicalClusterAL
from playground.sampling_methods.informative_diverse import (
    InformativeClusterDiverseSampler,
)


from playground.sampling_methods.kcenter_greedy import kCenterGreedy
from playground.sampling_methods.margin_AL import MarginAL
from playground.sampling_methods.mixture_of_samplers import MixtureOfSamplers
from playground.sampling_methods.represent_cluster_centers import (
    RepresentativeClusterMeanSampling,
)
from playground.sampling_methods.uniform_sampling import UniformSampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier

from framework_runners.alipy_runner import ALIPY_AL_Experiment
from framework_runners.optimal_runner import OPTIMAL_AL_Experiment
from framework_runners.libact_runner import LIBACT_Experiment
from framework_runners.playground_runner import PLAYGROUND_AL_Experiment
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
    ALIPY_UNCERTAINTY = (
        2  # ['least_confident', 'margin', 'entropy', 'distance_to_boundary']:
    )
    ALIPY_GRAPH_DENSITY = 3  # metric in ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',                      'braycurtis', 'canberra', 'chebyshev', 'correlation',                      'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',                      'russellrao', 'seuclidean', 'sokalmichener',                      'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]
    ALIPY_CORESET_GREEDY = (
        4  # distance in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
    )
    ALIPY_QUIRE = 5  # kernel=linear, poly, rbf
    OPTIMAL_BSO = 6
    OPTIMAL_TRUE = 7
    OPTIMAL_GREEDY = 8
    LIBACT_UNCERTAINTY = 9
    LIBACT_QUEY_BY_COMMITTEE = 10
    LIBACT_DWUS = 11
    LIBACT_QUIRE = 12
    LIBACT_VR = 13
    LIBACT_HINTSVM = 14
    PLAYGROUND_GRAPH_DENSITY = 15
    PLAYGROUND_HIERARCHICAL_CLUSTER = 16
    PLAYGROUND_INFORMATIVE_DIVERSE = 17
    PLAYGROUND_KCENTER_GREEDY = 18
    PLAYGROUND_MARGIN = 19
    PLAYGROUND_MIXTURE = 20
    PLAYGROUND_REPRESENTATIVE_CLUSTER = 21
    PLAYGROUND_UNIFORM = 22
    ALIPY_QBC = 23  # method='query_by_bagging' or 'vote_entropy'
    ALIPY_EXPECTED_ERROR_REDUCTION = 24
    ALIPY_BMDR = 25  # kernel
    ALIPY_SPAL = 26  # kernel
    ALIPY_LAL = 27  # mode: 'LAL_iterative', 'LAL_independent'
    ALIPY_DENSITY_WEIGHTED = 28  # uncertainty_meansure=['least_confident', 'margin', 'entropy'], distance=['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']


al_strategy_to_python_classes_mapping: Dict[AL_STRATEGY, Callable] = {
    AL_STRATEGY.ALIPY_RANDOM: QueryInstanceRandom,
    AL_STRATEGY.ALIPY_UNCERTAINTY: QueryInstanceUncertainty,
    AL_STRATEGY.ALIPY_GRAPH_DENSITY: QueryInstanceGraphDensity,
    AL_STRATEGY.ALIPY_CORESET_GREEDY: QueryInstanceCoresetGreedy,
    AL_STRATEGY.ALIPY_QUIRE: QueryInstanceQUIRE,
    AL_STRATEGY.ALIPY_QBC: QueryInstanceQBC,
    AL_STRATEGY.ALIPY_EXPECTED_ERROR_REDUCTION: QueryExpectedErrorReduction,
    AL_STRATEGY.ALIPY_BMDR: QueryInstanceBMDR,
    AL_STRATEGY.ALIPY_SPAL: QueryInstanceSPAL,
    AL_STRATEGY.ALIPY_LAL: QueryInstanceLAL,
    AL_STRATEGY.ALIPY_DENSITY_WEIGHTED: QueryInstanceDensityWeighted,
    AL_STRATEGY.OPTIMAL_GREEDY: Greedy_Optimal,
    AL_STRATEGY.OPTIMAL_BSO: Beeam_Search_Optimal,
    AL_STRATEGY.OPTIMAL_TRUE: True_Optimal,
    # AL_STRATEGY.OPTIMAL_SUBSETS: (, {}),
    AL_STRATEGY.LIBACT_UNCERTAINTY: UncertaintySampling,
    AL_STRATEGY.LIBACT_QUEY_BY_COMMITTEE: QueryByCommittee,
    AL_STRATEGY.LIBACT_DWUS: DWUS,
    AL_STRATEGY.LIBACT_QUIRE: QUIRE,
    AL_STRATEGY.PLAYGROUND_UNIFORM: UniformSampling,
    AL_STRATEGY.PLAYGROUND_MARGIN: MarginAL,
    AL_STRATEGY.PLAYGROUND_MIXTURE: MixtureOfSamplers,
    AL_STRATEGY.PLAYGROUND_KCENTER_GREEDY: kCenterGreedy,
    AL_STRATEGY.PLAYGROUND_REPRESENTATIVE_CLUSTER: RepresentativeClusterMeanSampling,
    AL_STRATEGY.PLAYGROUND_GRAPH_DENSITY: GraphDensitySampler,
    AL_STRATEGY.PLAYGROUND_HIERARCHICAL_CLUSTER: HierarchicalClusterAL,
    AL_STRATEGY.PLAYGROUND_INFORMATIVE_DIVERSE: InformativeClusterDiverseSampler,
}


def _import_compiled_libact_strategies():
    from libact.query_strategies import (
        HintSVM,
        VarianceReduction,
    )

    al_strategy_to_python_classes_mapping[AL_STRATEGY.LIBACT_VR] = VarianceReduction
    al_strategy_to_python_classes_mapping[AL_STRATEGY.LIBACT_HINTSVM] = HintSVM


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
    LOG_REG = 10
    SVM_LIBACT = 11


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
    LEARNER_MODEL.LOG_REG: (LogisticRegression, {}),
    LEARNER_MODEL.SVM_LIBACT: (
        SVM,
        {"kernal": "linear", "decision_function_shape": "ovr"},
    ),
}


@unique
class AL_FRAMEWORK(IntEnum):
    ALIPY = 1
    OPTIMAL = 2
    LIBACT = 3
    PLAYGROUND = 4


AL_framework_to_classes_mapping: Dict[AL_FRAMEWORK, Tuple[Callable, Dict[Any, Any]]] = {
    AL_FRAMEWORK.ALIPY: (ALIPY_AL_Experiment, {}),
    AL_FRAMEWORK.OPTIMAL: (OPTIMAL_AL_Experiment, {}),
    AL_FRAMEWORK.LIBACT: (LIBACT_Experiment, {}),
    AL_FRAMEWORK.PLAYGROUND: (PLAYGROUND_AL_Experiment, {}),
}
