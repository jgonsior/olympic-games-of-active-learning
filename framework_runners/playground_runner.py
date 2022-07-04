from __future__ import annotations
from typing import List
from misc.config import Config
from framework_runners.base_runner import AL_Experiment

class PLAYGROUND_AL_Experiment(AL_Experiment):
    def __init__(self, config: Config):
        super().__init__(config)
        self.al_strategy= None,

    def get_AL_strategy(self):
        from ressources.data_types import AL_STRATEGY
        from ressources.data_types import al_strategy_to_python_classes_mapping
        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy](self.X,self.Y,self.config.EXP_GRID_RANDOM_SEEDS_END)


    def query_AL_strategy(self) -> List[int]:
        pass

    def prepare_dataset(self):
        pass

