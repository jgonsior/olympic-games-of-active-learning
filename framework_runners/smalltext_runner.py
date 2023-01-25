from __future__ import annotations

from typing import TYPE_CHECKING
from framework_runners.base_runner import AL_Experiment
from small_text import LeastConfidence

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
    )


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
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](**additional_params)

    def query_AL_strategy(self) -> SampleIndiceList:
        if self.al_strategy is None:
            from misc.Errors import NoStrategyError
            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        if self.trn_ds is None:
            raise ValueError("prepare_dataset() has to be called before querying")
        return self.al_strategy.query(self.model, self.trn_ds, self.local_train_unlabeled_idx, self.local_train_labeled_idx, self.Y, self.config.EXP_BATCH_SIZE)

    def prepare_dataset(self):
        from small_text import SklearnDataset
        self.trn_ds = SklearnDataset(self.X, self.Y)
