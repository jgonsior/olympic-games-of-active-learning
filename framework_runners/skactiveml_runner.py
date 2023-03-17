from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from framework_runners.base_runner import AL_Experiment

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
    )

class SKACTIVEML_AL_Experiment(AL_Experiment):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.al_strategy = None
        self.density = None
        self.delta = 0.1
        self.switching_point = False
        self.u_max = -numpy.inf
        self.discriminator = None

    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import al_strategy_to_python_classes_mapping
        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        additional_params = al_strategy_to_python_classes_mapping[strategy][1]
        additional_params["random_state"] = self.config.RANDOM_SEED
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](**additional_params)
        if self.config.EXP_STRATEGY in (AL_STRATEGY.SKACTIVEML_DWUS, AL_STRATEGY.SKACTIVEML_DUAL_STRAT):
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(init_params="kmeans", n_components=5)
            gmm.fit(self.local_X_train)
            self.density = numpy.exp(gmm.score_samples(self.local_X_train))
        elif self.config.EXP_STRATEGY == AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL:
            from sklearn import clone
            self.discriminator = clone(self.model)
        elif self.config.EXP_STRATEGY == AL_STRATEGY.SKACTIVEML_MCPAL:
            dens_est = self.model.fit(self.local_X_train, numpy.zeros(len(self.local_X_train)))
            self.density = self.model.predict_freq(self.local_X_train)[:, 0]

    def query_AL_strategy(self) -> SampleIndiceList:
        from resources.data_types import AL_STRATEGY
        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        ret = []
        match self.config.EXP_STRATEGY:
            case (AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_OUTPUT_CHANGE |
                  AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_VARIANCE_REDUCTION |
                  AL_STRATEGY.SKACTIVEML_KL_DIVERGENCE_MAXIMIZATION |
                  AL_STRATEGY.SKACTIVEML_EXPECTED_MODEL_CHANGE |
                  AL_STRATEGY.SKACTIVEML_GREEDY_TARGET_SPACE |
                  AL_STRATEGY.SKACTIVEML_GREEDY_IMPROVED):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, reg=self.model)
            case (AL_STRATEGY.SKACTIVEML_MC_EER_LOG_LOSS |
                  AL_STRATEGY.SKACTIVEML_MC_EER_MISCLASS_LOSS |
                  AL_STRATEGY.SKACTIVEML_VOI_UNLABELED |
                  AL_STRATEGY.SKACTIVEML_VOI_LABELED |
                  AL_STRATEGY.SKACTIVEML_VOI):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, clf=self.model, ignore_partial_fit=True)
            case (AL_STRATEGY.LIBACT_QBC |
                  AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, ensemble=self.model)
            case (AL_STRATEGY.SKACTIVEML_EPISTEMIC_US |
                  AL_STRATEGY.SKACTIVEML_DDDD |
                  AL_STRATEGY.SKACTIVEML_US_MARGIN |
                  AL_STRATEGY.SKACTIVEML_US_LC |
                  AL_STRATEGY.SKACTIVEML_US_ENTROPY |
                  AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, clf=self.model)
            case (AL_STRATEGY.SKACTIVEML_DWUS |
                  AL_STRATEGY.SKACTIVEML_MCPAL):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, clf=self.model, utility_weight=self.density)
            case AL_STRATEGY.SKACTIVEML_DUAL_STRAT:
                from skactiveml.utils import simple_batch
                if not self.switching_point:
                    ret, utils = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, clf=self.model, utility_weight=self.density, return_utilities=True)
                    utilities = utils[0]
                    self.switching_point = utilities[ret[0]]-self.u_max < self.delta
                    self.u_max = utilities[ret[0]]
                else:
                    utils_US = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, clf=self.model, return_utilities=True)[1][0]
                    err = numpy.nanmean(utils_US)
                    utilities = (1 - err) * utils_US + err * self.density
                    ret = simple_batch(utilities, self.config.RANDOM_SEED)
            case (AL_STRATEGY.SKACTIVEML_COST_EMBEDDING |
                  AL_STRATEGY.SKACTIVEML_GREEDY_FEATURE_SPACE |
                  AL_STRATEGY.SKACTIVEML_QUIRE):
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train)
            case AL_STRATEGY.SMALLTEXT_DISCRIMINATIVEAL:
                ret = self.al_strategy.query(X=self.local_X_train, Y=self.local_Y_train, discriminator=self.discriminator)
        return None

    def prepare_dataset(self):
        pass