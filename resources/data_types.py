from __future__ import annotations
import multiprocessing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple
import numpy as np
from enum import IntEnum, unique
from alipy.query_strategy import (
    QueryInstanceUncertainty,
    QueryInstanceRandom,
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
    ActiveLearningByLearning,
)
from libact.query_strategies.multiclass import EER, HierarchicalSampling

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
from framework_runners.smalltext_runner import SMALLTEXT_AL_Experiment
from framework_runners.skactiveml_runner import SKACTIVEML_AL_Experiment

from metrics.computed.CLASS_DISTRIBUTIONS import CLASS_DISTRIBUTIONS
from metrics.computed.DATASET_CATEGORIZATION import DATASET_CATEGORIZATION
from metrics.computed.DISTANCE_METRICS import DISTANCE_METRICS
from metrics.computed.METRIC_DROP import METRIC_DROP
from metrics.computed.MISMATCH_TRAIN_TEST import MISMATCH_TRAIN_TEST
from metrics.computed.STANDARD_AUC import STANDARD_AUC
from metrics.computed.TIMELAG_METRIC import TIMELAG_METRIC
from metrics.computed.base_samples_categorizer import (
    AVERAGE_UNCERTAINTY,
    CLOSENESS_TO_CLUSTER_CENTER,
    CLOSENESS_TO_DECISION_BOUNDARY,
    CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS,
    CLOSENESS_TO_SAMPLES_OF_SAME_CLASS,
    COUNT_WRONG_CLASSIFICATIONS,
    IMPROVES_ACCURACY_BY,
    INCLUDED_IN_OPTIMAL_STRATEGY,
    MELTING_POT_REGION,
    OUTLIERNESS,
    REGION_DENSITY,
    SWITCHES_CLASS_OFTEN,
    CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN,
    CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN,
)
from optimal_query_strategies.BSO_optimal import Beeam_Search_Optimal

from small_text import (
    LeastConfidence,
    PredictionEntropy,
    BreakingTies,
    EmbeddingKMeans,
    GreedyCoreset,
    LightweightCoreset,
    ContrastiveActiveLearning,
    DiscriminativeActiveLearning,
    CategoryVectorInconsistencyAndRanking,
    # SEALS,
    # BALD,
    RandomSampling,
)

from skactiveml.pool import (
    # tChange,
    ExpectedModelVarianceReduction,
    KLDivergenceMaximization,
    MonteCarloEER,
    ValueOfInformationEER,
    ExpectedModelChangeMaximization,
    QueryByCommittee as SK_QueryByCommittee,
    # EpistemicUncertaintySampling,
    FourDs,
    UncertaintySampling as SK_UncertaintySampling,
    CostEmbeddingAL,
    DiscriminativeAL,
    GreedySamplingTarget,
    GreedySamplingX,
    ProbabilisticAL,
    Quire,
)

from optimal_query_strategies.greedy_optimal import (
    FuturePeakEvalMetric,
    Greedy_Optimal,
)
from optimal_query_strategies.true_optimal import True_Optimal

if TYPE_CHECKING:
    pass

SampleIndiceList = List[int]
FeatureVectors = np.ndarray
LabelList = np.ndarray


@unique
class AL_STRATEGY(IntEnum):
    ALIPY_RANDOM = 1
    ALIPY_UNCERTAINTY_LC = 2
    ALIPY_GRAPH_DENSITY = 3  # metric in ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',                      'braycurtis', 'canberra', 'chebyshev', 'correlation',                      'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',                      'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',                      'russellrao', 'seuclidean', 'sokalmichener',                      'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]
    ALIPY_CORESET_GREEDY = (
        4  # distance in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'].
    )
    # ALIPY_QUIRE = 5  # kernel=linear, poly, rbf -> coverd by LIBACT_QUIRE
    OPTIMAL_BSO = 6
    OPTIMAL_TRUE = 7
    OPTIMAL_GREEDY_10 = 8
    LIBACT_UNCERTAINTY_LC = 9
    LIBACT_QBC = 10
    LIBACT_DWUS = 11
    LIBACT_QUIRE = 12
    PLAYGROUND_GRAPH_DENSITY = 15
    PLAYGROUND_HIERARCHICAL_CLUSTER = 16
    PLAYGROUND_INFORMATIVE_DIVERSE = 17
    PLAYGROUND_KCENTER_GREEDY = 18
    PLAYGROUND_MARGIN = 19
    PLAYGROUND_MIXTURE = 20
    PLAYGROUND_MCM = 21
    PLAYGROUND_UNIFORM = 22
    ALIPY_QBC = 23  # method='query_by_bagging' or 'vote_entropy'
    ALIPY_EXPECTED_ERROR_REDUCTION = 24
    ALIPY_BMDR = 25  # kernel
    ALIPY_SPAL = 26  # kernel
    ALIPY_LAL = 27  # mode: 'LAL_iterative', 'LAL_independent'
    ALIPY_DENSITY_WEIGHTED = 28  # uncertainty_meansure=['least_confident', 'margin', 'entropy'], distance=['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    LIBACT_EER = 29
    LIBACT_HIERARCHICAL_SAMPLING = 30
    LIBACT_ALBL = 31
    PLAYGROUND_BANDIT = 32
    ALIPY_UNCERTAINTY_MM = 33
    ALIPY_UNCERTAINTY_ENTROPY = 34
    ALIPY_UNCERTAINTY_DTB = 35
    LIBACT_UNCERTAINTY_SM = 36
    LIBACT_UNCERTAINTY_ENT = 37
    OPTIMAL_GREEDY_20 = 38
    SMALLTEXT_LEASTCONFIDENCE = 39
    SMALLTEXT_PREDICTIONENTROPY = 40
    SMALLTEXT_BREAKINGTIES = 41
    # SMALLTEXT_BALD = 42
    SMALLTEXT_EMBEDDINGKMEANS = 43
    SMALLTEXT_GREEDYCORESET = 44
    SMALLTEXT_LIGHTWEIGHTCORESET = 45
    SMALLTEXT_CONTRASTIVEAL = 46
    SMALLTEXT_DISCRIMINATIVEAL = 47
    # SMALLTEXT_CVIAR = 48  # Category Vector Inconsistency and Ranking
    # SMALLTEXT_SEALS = 49
    SMALLTEXT_RANDOM = 50
    # SKACTIVEML_EXPECTED_MODEL_OUTPUT_CHANGE = 51
    # SKACTIVEML_EXPECTED_MODEL_VARIANCE_REDUCTION = 52
    # SKACTIVEML_KL_DIVERGENCE_MAXIMIZATION = 53
    SKACTIVEML_MC_EER_LOG_LOSS = 54
    SKACTIVEML_MC_EER_MISCLASS_LOSS = 55
    SKACTIVEML_VOI_UNLABELED = 56
    SKACTIVEML_VOI_LABELED = 57
    SKACTIVEML_VOI = 58
    # = 59
    SKACTIVEML_QBC = 60
    # SKACTIVEML_EPISTEMIC_US = 61
    # SKACTIVEML_DDDD = 62
    SKACTIVEML_US_MARGIN = 63
    SKACTIVEML_US_LC = 64
    SKACTIVEML_US_ENTROPY = 65
    SKACTIVEML_EXPECTED_AVERAGE_PRECISION = 66
    SKACTIVEML_DWUS = 67
    # SKACTIVEML_DUAL_STRAT = 68
    SKACTIVEML_COST_EMBEDDING = 69
    SKACTIVEML_DAL = 70
    # SKACTIVEML_GREEDY_TARGET_SPACE = 71
    # SKACTIVEML_GREEDY_IMPROVED = 72
    # SKACTIVEML_GREEDY_FEATURE_SPACE = 73
    SKACTIVEML_MCPAL = 74
    SKACTIVEML_QBC_VOTE_ENTROPY = 75
    SKACTIVEML_QUIRE = 76


al_strategy_to_python_classes_mapping: Dict[
    AL_STRATEGY, Tuple[Callable, Dict[Any, Any]]
] = {
    AL_STRATEGY.ALIPY_RANDOM: (QueryInstanceRandom, {}),
    AL_STRATEGY.ALIPY_UNCERTAINTY_LC: (
        QueryInstanceUncertainty,
        {"measure": "least_confident"},
    ),
    AL_STRATEGY.ALIPY_UNCERTAINTY_MM: (QueryInstanceUncertainty, {"measure": "margin"}),
    AL_STRATEGY.ALIPY_UNCERTAINTY_ENTROPY: (
        QueryInstanceUncertainty,
        {"measure": "entropy"},
    ),
    AL_STRATEGY.ALIPY_UNCERTAINTY_DTB: (
        QueryInstanceUncertainty,
        {"measure": "distance_to_boundary"},
    ),
    AL_STRATEGY.ALIPY_GRAPH_DENSITY: (QueryInstanceGraphDensity, {}),
    AL_STRATEGY.ALIPY_CORESET_GREEDY: (QueryInstanceCoresetGreedy, {}),
    # AL_STRATEGY.ALIPY_QUIRE: (QueryInstanceQUIRE, {}),
    AL_STRATEGY.ALIPY_QBC: (QueryInstanceQBC, {}),
    AL_STRATEGY.ALIPY_EXPECTED_ERROR_REDUCTION: (QueryExpectedErrorReduction, {}),
    AL_STRATEGY.ALIPY_BMDR: (QueryInstanceBMDR, {}),
    AL_STRATEGY.ALIPY_SPAL: (QueryInstanceSPAL, {}),
    AL_STRATEGY.ALIPY_LAL: (QueryInstanceLAL, {"train_slt": False}),
    AL_STRATEGY.ALIPY_DENSITY_WEIGHTED: (QueryInstanceDensityWeighted, {}),
    AL_STRATEGY.OPTIMAL_GREEDY_10: (
        Greedy_Optimal,
        {
            "amount_of_pre_selections": 10,
            "future_peak_eval_metric": FuturePeakEvalMetric.ACC,
        },
    ),
    AL_STRATEGY.OPTIMAL_GREEDY_20: (
        Greedy_Optimal,
        {
            "amount_of_pre_selections": 20,
            "future_peak_eval_metric": FuturePeakEvalMetric.ACC,
        },
    ),
    AL_STRATEGY.OPTIMAL_BSO: (
        Beeam_Search_Optimal,
        {"future_peak_eval_metric": FuturePeakEvalMetric.ACC},
    ),
    AL_STRATEGY.OPTIMAL_TRUE: (
        True_Optimal,
        {"future_peak_eval_metric": FuturePeakEvalMetric.ACC},
    ),
    # AL_STRATEGY.OPTIMAL_SUBSETS: (, {}),
    AL_STRATEGY.LIBACT_UNCERTAINTY_LC: (UncertaintySampling, {"method": "lc"}),
    AL_STRATEGY.LIBACT_UNCERTAINTY_SM: (UncertaintySampling, {"method": "sm"}),
    AL_STRATEGY.LIBACT_UNCERTAINTY_ENT: (UncertaintySampling, {"method": "entropy"}),
    AL_STRATEGY.LIBACT_QBC: (QueryByCommittee, {}),
    AL_STRATEGY.LIBACT_DWUS: (DWUS, {}),
    AL_STRATEGY.LIBACT_QUIRE: (QUIRE, {}),
    AL_STRATEGY.LIBACT_EER: (EER, {}),
    AL_STRATEGY.LIBACT_HIERARCHICAL_SAMPLING: (HierarchicalSampling, {}),
    AL_STRATEGY.LIBACT_ALBL: (ActiveLearningByLearning, {}),
    AL_STRATEGY.PLAYGROUND_UNIFORM: (UniformSampling, {}),
    AL_STRATEGY.PLAYGROUND_MARGIN: (MarginAL, {}),
    AL_STRATEGY.PLAYGROUND_MIXTURE: (MixtureOfSamplers, {}),
    AL_STRATEGY.PLAYGROUND_KCENTER_GREEDY: (kCenterGreedy, {}),
    AL_STRATEGY.PLAYGROUND_MCM: (RepresentativeClusterMeanSampling, {}),
    AL_STRATEGY.PLAYGROUND_GRAPH_DENSITY: (GraphDensitySampler, {}),
    AL_STRATEGY.PLAYGROUND_HIERARCHICAL_CLUSTER: (HierarchicalClusterAL, {}),
    AL_STRATEGY.PLAYGROUND_INFORMATIVE_DIVERSE: (InformativeClusterDiverseSampler, {}),
    AL_STRATEGY.PLAYGROUND_BANDIT: (BanditDiscreteSampler, {}),
    AL_STRATEGY.SMALLTEXT_LEASTCONFIDENCE: (LeastConfidence, {}),
    AL_STRATEGY.SMALLTEXT_PREDICTIONENTROPY: (PredictionEntropy, {}),
    AL_STRATEGY.SMALLTEXT_BREAKINGTIES: (BreakingTies, {}),
    # AL_STRATEGY.SMALLTEXT_BALD: (BALD, {}),
    AL_STRATEGY.SMALLTEXT_EMBEDDINGKMEANS: (EmbeddingKMeans, {}),
    AL_STRATEGY.SMALLTEXT_GREEDYCORESET: (GreedyCoreset, {}),
    AL_STRATEGY.SMALLTEXT_LIGHTWEIGHTCORESET: (LightweightCoreset, {}),
    AL_STRATEGY.SMALLTEXT_CONTRASTIVEAL: (ContrastiveActiveLearning, {}),
    AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL: (DiscriminativeActiveLearning, {}),
    # AL_STRATEGY.SMALLTEXT_CVIAR: (CategoryVectorInconsistencyAndRanking, {}),
    # AL_STRATEGY.SMALLTEXT_SEALS: (SEALS, {}),
    AL_STRATEGY.SMALLTEXT_RANDOM: (RandomSampling, {}),
    # AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_OUTPUT_CHANGE: (
    #    ExpectedModelOutputChange,
    #    {"integration_dict": {"method": "gauss_hermite", "n_integration_samples": 5}},
    # ),
    # AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_VARIANCE_REDUCTION: (
    #    ExpectedModelVarianceReduction,
    #    {},
    # ),
    # AL_STRATEGY.SKACTIVEML_KL_DIVERGENCE_MAXIMIZATION: (
    #    KLDivergenceMaximization,
    #    {
    #        "integration_dict_target_val": {
    #            "method": "gauss_hermite",
    #            "n_integration_samples": 5,
    #        }
    #    },
    # ),
    AL_STRATEGY.SKACTIVEML_MC_EER_LOG_LOSS: (MonteCarloEER, {"method": "log_loss"}),
    AL_STRATEGY.SKACTIVEML_MC_EER_MISCLASS_LOSS: (MonteCarloEER, {}),
    AL_STRATEGY.SKACTIVEML_VOI_UNLABELED: (
        ValueOfInformationEER,
        {"consider_labeled": False, "subtract_current": True, "normalize": True},
    ),
    AL_STRATEGY.SKACTIVEML_VOI_LABELED: (
        ValueOfInformationEER,
        {"consider_unlabeled": False, "candidate_to_labeled": False},
    ),
    AL_STRATEGY.SKACTIVEML_VOI: (ValueOfInformationEER, {"subtract_current": True}),
    # AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_CHANGE: (ExpectedModelChangeMaximization, {}),
    AL_STRATEGY.SKACTIVEML_QBC: (SK_QueryByCommittee, {}),
    # AL_STRATEGY.SKACTIVEML_EPISTEMIC_US: (EpistemicUncertaintySampling, {}),
    # AL_STRATEGY.SKACTIVEML_DDDD: (FourDs, {}),
    AL_STRATEGY.SKACTIVEML_US_MARGIN: (
        SK_UncertaintySampling,
        {"method": "margin_sampling"},
    ),
    AL_STRATEGY.SKACTIVEML_US_LC: (SK_UncertaintySampling, {}),
    AL_STRATEGY.SKACTIVEML_US_ENTROPY: (SK_UncertaintySampling, {"method": "entropy"}),
    AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION: (
        SK_UncertaintySampling,
        {"method": "expected_average_precision"},
    ),
    AL_STRATEGY.SKACTIVEML_DWUS: (SK_UncertaintySampling, {}),
    # AL_STRATEGY.SKACTIVEML_DUAL_STRAT: (SK_UncertaintySampling, {}),
    AL_STRATEGY.SKACTIVEML_COST_EMBEDDING: (CostEmbeddingAL, {}),
    AL_STRATEGY.SKACTIVEML_DAL: (DiscriminativeAL, {}),
    # AL_STRATEGY.SKACTIVEML_GREEDY_TARGET_SPACE: (
    #    GreedySamplingTarget,
    #    {"method": "GSy"},
    # ),
    # AL_STRATEGY.SKACTIVEML_GREEDY_IMPROVED: (GreedySamplingTarget, {}),
    # AL_STRATEGY.SKACTIVEML_GREEDY_FEATURE_SPACE: (GreedySamplingX, {}),
    AL_STRATEGY.SKACTIVEML_MCPAL: (ProbabilisticAL, {}),
    AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY: (
        SK_QueryByCommittee,
        {"method": "vote_entropy"},
    ),
    AL_STRATEGY.SKACTIVEML_QUIRE: (
        Quire,
        {"classes": [0, 1], "metric_dict": {"gamma": 0.1}},
    ),
}


al_strategies_which_only_support_binary_classification: List[AL_STRATEGY] = [
    AL_STRATEGY.ALIPY_LAL,
    AL_STRATEGY.ALIPY_UNCERTAINTY_DTB,
    AL_STRATEGY.ALIPY_BMDR,
    AL_STRATEGY.ALIPY_SPAL,
    AL_STRATEGY.LIBACT_HINTSVM,
]

al_strategies_which_require_decision_boundary_model: List[AL_STRATEGY] = [
    AL_STRATEGY.PLAYGROUND_MCM,
    AL_STRATEGY.ALIPY_UNCERTAINTY_DTB,
]

al_strategies_not_suitable_for_hpc: List[AL_STRATEGY] = [
    AL_STRATEGY.ALIPY_LAL,
]


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
    LR = 12


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
            "learning_rate": "adaptive",
            "max_iter": 1000,
            "warm_start": False,
            "early_stopping": False,
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
    LEARNER_MODEL.LR: (LogisticRegression, {}),
}


@unique
class AL_FRAMEWORK(IntEnum):
    ALIPY = 1
    OPTIMAL = 2
    LIBACT = 3
    PLAYGROUND = 4
    SMALLTEXT = 5
    SKACTIVEML = 6


AL_framework_to_classes_mapping: Dict[AL_FRAMEWORK, Tuple[Callable, Dict[Any, Any]]] = {
    AL_FRAMEWORK.ALIPY: (ALIPY_AL_Experiment, {}),
    AL_FRAMEWORK.OPTIMAL: (OPTIMAL_AL_Experiment, {}),
    AL_FRAMEWORK.LIBACT: (LIBACT_Experiment, {}),
    AL_FRAMEWORK.PLAYGROUND: (PLAYGROUND_AL_Experiment, {}),
    AL_FRAMEWORK.SMALLTEXT: (SMALLTEXT_AL_Experiment, {}),
    AL_FRAMEWORK.SKACTIVEML: (SKACTIVEML_AL_Experiment, {}),
}


@unique
class COMPUTED_METRIC(IntEnum):
    STANDARD_AUC = 1
    DISTANCE_METRICS = 2
    MISMATCH_TRAIN_TEST = 3
    CLASS_DISTRIBUTIONS = 4
    METRIC_DROP = 5
    DATASET_CATEGORIZATION = 6
    TIMELAG_METRIC = 7


computed_metric_to_classes_mapping: Dict[COMPUTED_METRIC, Callable] = {
    COMPUTED_METRIC.STANDARD_AUC: STANDARD_AUC,
    COMPUTED_METRIC.DISTANCE_METRICS: DISTANCE_METRICS,
    COMPUTED_METRIC.MISMATCH_TRAIN_TEST: MISMATCH_TRAIN_TEST,
    COMPUTED_METRIC.CLASS_DISTRIBUTIONS: CLASS_DISTRIBUTIONS,
    COMPUTED_METRIC.METRIC_DROP: METRIC_DROP,
    COMPUTED_METRIC.DATASET_CATEGORIZATION: DATASET_CATEGORIZATION,
    COMPUTED_METRIC.TIMELAG_METRIC: TIMELAG_METRIC,
}


@unique
class SAMPLES_CATEGORIZER(IntEnum):
    COUNT_WRONG_CLASSIFICATIONS = 1
    SWITCHES_CLASS_OFTEN = 2
    CLOSENESS_TO_DECISION_BOUNDARY = 3
    REGION_DENSITY = 4
    MELTING_POT_REGION = 5
    INCLUDED_IN_OPTIMAL_STRATEGY = 6
    CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN = 7
    CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN = 8
    CLOSENESS_TO_CLUSTER_CENTER = 9
    IMPROVES_ACCURACY_BY = 10
    AVERAGE_UNCERTAINTY = 11
    OUTLIERNESS = 12
    CLOSENESS_TO_SAMPLES_OF_SAME_CLASS = 13
    CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS = 14


samples_categorizer_to_classes_mapping: Dict[SAMPLES_CATEGORIZER, Callable] = {
    SAMPLES_CATEGORIZER.COUNT_WRONG_CLASSIFICATIONS: COUNT_WRONG_CLASSIFICATIONS,
    SAMPLES_CATEGORIZER.SWITCHES_CLASS_OFTEN: SWITCHES_CLASS_OFTEN,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_DECISION_BOUNDARY: CLOSENESS_TO_DECISION_BOUNDARY,
    SAMPLES_CATEGORIZER.REGION_DENSITY: REGION_DENSITY,
    SAMPLES_CATEGORIZER.MELTING_POT_REGION: MELTING_POT_REGION,
    SAMPLES_CATEGORIZER.INCLUDED_IN_OPTIMAL_STRATEGY: INCLUDED_IN_OPTIMAL_STRATEGY,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_SAMPLES_OF_SAME_CLASS: CLOSENESS_TO_SAMPLES_OF_SAME_CLASS,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS: CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN: CLOSENESS_TO_SAMPLES_OF_SAME_CLASS_kNN,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN: CLOSENESS_TO_SAMPLES_OF_OTHER_CLASS_kNN,
    SAMPLES_CATEGORIZER.CLOSENESS_TO_CLUSTER_CENTER: CLOSENESS_TO_CLUSTER_CENTER,
    SAMPLES_CATEGORIZER.IMPROVES_ACCURACY_BY: IMPROVES_ACCURACY_BY,
    SAMPLES_CATEGORIZER.AVERAGE_UNCERTAINTY: AVERAGE_UNCERTAINTY,
    SAMPLES_CATEGORIZER.OUTLIERNESS: OUTLIERNESS,
}
