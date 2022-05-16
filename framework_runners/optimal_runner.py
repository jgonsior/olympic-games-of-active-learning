from __future__ import annotations

from typing import TYPE_CHECKING, List
from framework_runners.base_runner import AL_Experiment


class OPTIMAL_AL_Experiment(AL_Experiment):
    def get_AL_strategy(self):
        from ressources.data_types import AL_STRATEGY
        from ressources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        if self.config.EXP_STRATEGY == AL_STRATEGY.OPTIMAL_BSO:
            self.config.EXP_STRATEGY_PARAMS["num_queries"] = self.config.EXP_NUM_QUERIES

        al_strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        print(al_strategy)
        al_strategy = al_strategy_to_python_classes_mapping[al_strategy](
            X=self.X,
            Y=self.Y,
            train_idx=self.train_idx,
            **self.config.EXP_STRATEGY_PARAMS,
        )
        self.al_strategy = al_strategy

    def query_AL_strategy(self) -> List[int]:
        select_ind = self.al_strategy.select(
            labeled_index=self.label_idx,
            unlabeled_index=self.unlabel_idx,
            model=self.model,
            batch_size=self.config.EXP_BATCH_SIZE,
        )
        if not isinstance(select_ind, list):
            select_ind = select_ind.tolist()
        return select_ind

    # dataset in numpy format and indice lists are fine as it is
    def prepare_dataset(self):
        pass
