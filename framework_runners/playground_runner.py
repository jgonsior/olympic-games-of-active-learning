from __future__ import annotations
from typing import List
from misc.config import Config
from framework_runners.base_runner import AL_Experiment

class PLAYGROUND_AL_Experiment(AL_Experiment):
    def __init__(self, config: Config):
        super().__init__(config)
        self.al_strategy = None,

    def get_AL_strategy(self):
        from ressources.data_types import AL_STRATEGY
        from ressources.data_types import al_strategy_to_python_classes_mapping
        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy](self.X, self.Y, self.config.RANDOM_SEED)

    def query_AL_strategy(self) -> List[int]:
        if self.al_strategy is None:
            raise RuntimeError("get_AL_strategy() has to be called before querying")
        return self.al_strategy.select_batch_(self.model, self.label_idx, self.config.EXP_BATCH_SIZE)

    def prepare_dataset(self):
        pass

