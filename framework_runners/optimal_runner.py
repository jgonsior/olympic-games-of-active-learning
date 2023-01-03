from __future__ import annotations


from typing import TYPE_CHECKING

from framework_runners.base_runner import AL_Experiment

if TYPE_CHECKING:
    pass

    from resources.data_types import (
        SampleIndiceList,
    )


class OPTIMAL_AL_Experiment(AL_Experiment):
    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        al_strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        additional_params = al_strategy_to_python_classes_mapping[al_strategy][1]

        if self.config.EXP_STRATEGY == AL_STRATEGY.OPTIMAL_BSO:
            additional_params["num_queries"] = self.config.EXP_NUM_QUERIES

        al_strategy = al_strategy_to_python_classes_mapping[al_strategy][0](
            X=self.local_X_train,
            Y=self.local_Y_train,
            **additional_params,
        )
        self.al_strategy = al_strategy

    def query_AL_strategy(self) -> SampleIndiceList:
        select_ind = self.al_strategy.select(
            labeled_index=self.local_train_labeled_idx,
            unlabeled_index=self.local_train_unlabeled_idx,
            model=self.model,
            batch_size=self.config.EXP_BATCH_SIZE,
        )
        if not isinstance(select_ind, list):
            select_ind = select_ind.tolist()
        return select_ind

    # dataset in numpy format and indice lists are fine as it is
    def prepare_dataset(self):
        pass
