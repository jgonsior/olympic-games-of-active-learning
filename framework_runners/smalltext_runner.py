from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from framework_runners.base_runner import AL_Experiment
from small_text import SklearnClassifier, PoolBasedActiveLearner

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import SampleIndiceList


class SMALLTEXT_AL_Experiment(AL_Experiment):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.trn_ds = None
        self.al_strategy = None

    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import al_strategy_to_python_classes_mapping

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        additional_params = al_strategy_to_python_classes_mapping[strategy][1]

        query_strategy = al_strategy_to_python_classes_mapping[strategy][0](
            **additional_params
        )
        self._small_text_model = SklearnClassifier(
            self.model,
            num_classes=np.unique(self.Y),
        )

        self.al_strategy = query_strategy

    def query_AL_strategy(self) -> SampleIndiceList:
        from resources.data_types import AL_STRATEGY

        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        if self.trn_ds is None:
            raise ValueError("prepare_dataset() has to be called before querying")

        if self.config.EXP_STRATEGY == AL_STRATEGY.SMALLTEXT_RANDOM:
            queries = self.al_strategy.query(
                clf=self._small_text_model,
                _dataset=self.trn_ds,
                indices_unlabeled=self.local_train_unlabeled_idx,
                indices_labeled=self.local_train_labeled_idx,
                y=self.local_Y_train[self.local_train_labeled_idx],
                n=self.config.EXP_BATCH_SIZE,
            )
        else:
            queries = self.al_strategy.query(
                clf=self._small_text_model,
                dataset=self.trn_ds,
                indices_unlabeled=self.local_train_unlabeled_idx,
                indices_labeled=self.local_train_labeled_idx,
                y=self.local_Y_train[self.local_train_labeled_idx],
                n=self.config.EXP_BATCH_SIZE,
            )
        return queries.tolist()

    def prepare_dataset(self):
        from small_text import SklearnDataset

        self.trn_ds = SklearnDataset(
            self.local_X_train, self.local_Y_train, target_labels=np.unique(self.Y)
        )
