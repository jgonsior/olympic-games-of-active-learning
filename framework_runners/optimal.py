from __future__ import annotations

from typing import TYPE_CHECKING, List
from framework_runners.base_runner import AL_Experiment

if TYPE_CHECKING:
    from ressources.data_types import AL_STRATEGY
    from ressources.data_types import (
        al_strategy_to_python_classes_mapping,
    )


class OPTIMAL_AL_Experiment(AL_Experiment):
    def get_AL_strategy(self):
        al_strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        al_strategy = al_strategy_to_python_classes_mapping[al_strategy][0](
            X=self.X, Y=self.Y, **al_strategy_to_python_classes_mapping[al_strategy][1]
        )
        self.al_strategy = al_strategy

    def query_AL_strategy(self) -> List[int]:
        select_ind = self.al_strategy.select(
            label_index=self.label_idx,
            unlabel_index=self.unlabel_idx,
            model=self.model,
            batch_size=self.config.EXP_BATCH_SIZE,
        )
        if not isinstance(select_ind, list):
            select_ind = select_ind.tolist()
        return select_ind

    # dataset in numpy format and indice lists are fine as it is
    def prepare_dataset(self):
        pass
