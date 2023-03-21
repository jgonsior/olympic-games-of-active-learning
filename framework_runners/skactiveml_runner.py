from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import numpy as np

from framework_runners.base_runner import AL_Experiment
from skactiveml.classifier import SklearnClassifier
from skactiveml.utils import MISSING_LABEL

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

        self.model = SklearnClassifier(self.model, classes=numpy.unique(self.Y))

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        additional_params = al_strategy_to_python_classes_mapping[strategy][1]
        additional_params["random_state"] = self.config.RANDOM_SEED
        if self.config.EXP_STRATEGY in (AL_STRATEGY.SKACTIVEML_DWUS):
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(init_params="kmeans", n_components=5)
            gmm.fit(self.local_X_train)
            self.density = numpy.exp(gmm.score_samples(self.local_X_train))
        elif self.config.EXP_STRATEGY == AL_STRATEGY.SKACTIVEML_DAL:
            from sklearn import clone

            self.discriminator = clone(self.model)
        elif self.config.EXP_STRATEGY == AL_STRATEGY.SKACTIVEML_MCPAL:
            dens_est = self.model.fit(
                self.local_X_train, numpy.zeros(len(self.local_X_train))
            )
            self.density = self.model.predict_freq(self.local_X_train)[:, 0]
        elif (
            self.config.EXP_STRATEGY == AL_STRATEGY.SKACTIVEML_QUIRE
            or self.config.EXP_STRATEGY == AL_STRATEGY.SKACTIVEML_COST_EMBEDDING
        ):
            additional_params["classes"] = np.unique(self.Y)

        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](
            **additional_params
        )

    def query_AL_strategy(self) -> SampleIndiceList:
        from resources.data_types import AL_STRATEGY

        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        ret = []

        y_with_nans = numpy.full(
            shape=self.local_Y_train.shape, fill_value=MISSING_LABEL
        )
        y_with_nans[self.local_train_labeled_idx] = self.local_Y_train[
            self.local_train_labeled_idx
        ]

        # self.model.fit(self.local_X_train, y_with_nans)

        match self.config.EXP_STRATEGY:
            case (
                AL_STRATEGY.SKACTIVEML_MC_EER_LOG_LOSS
                | AL_STRATEGY.SKACTIVEML_MC_EER_MISCLASS_LOSS
                | AL_STRATEGY.SKACTIVEML_VOI_UNLABELED
                | AL_STRATEGY.SKACTIVEML_VOI_LABELED
                | AL_STRATEGY.SKACTIVEML_VOI
            ):
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    clf=self.model,
                    ignore_partial_fit=True,
                    batch_size=self.config.EXP_BATCH_SIZE,
                )
            case (AL_STRATEGY.SKACTIVEML_QBC | AL_STRATEGY.SKACTIVEML_QBC_VOTE_ENTROPY):
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    ensemble=[
                        self.model,
                        self.model,
                        self.model,
                        self.model,
                        self.model,
                    ],
                    batch_size=self.config.EXP_BATCH_SIZE,
                )
            case (
                AL_STRATEGY.SKACTIVEML_US_MARGIN
                | AL_STRATEGY.SKACTIVEML_US_LC
                | AL_STRATEGY.SKACTIVEML_US_ENTROPY
                | AL_STRATEGY.SKACTIVEML_EXPECTED_AVERAGE_PRECISION
            ):
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    clf=self.model,
                    batch_size=self.config.EXP_BATCH_SIZE,
                )
            case (AL_STRATEGY.SKACTIVEML_DWUS | AL_STRATEGY.SKACTIVEML_MCPAL):
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    clf=self.model,
                    utility_weight=self.density,
                    batch_size=self.config.EXP_BATCH_SIZE,
                )
            case (AL_STRATEGY.SKACTIVEML_COST_EMBEDDING | AL_STRATEGY.SKACTIVEML_QUIRE):
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    batch_size=self.config.EXP_BATCH_SIZE,
                )
            case AL_STRATEGY.SKACTIVEML_DAL:
                ret = self.al_strategy.query(
                    X=self.local_X_train,
                    y=y_with_nans,
                    discriminator=self.discriminator,
                    batch_size=self.config.EXP_BATCH_SIZE,
                )

        return ret.tolist()

    def prepare_dataset(self):
        pass
