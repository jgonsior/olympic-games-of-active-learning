from __future__ import annotations

from framework_runners.base_runner import AL_Experiment

from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
    )


class PLAYGROUND_AL_Experiment(AL_Experiment):
    def __init__(self, config: Config):
        super().__init__(config)
        self.al_strategy = (None,)

    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import al_strategy_to_python_classes_mapping

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)

        additional_params = al_strategy_to_python_classes_mapping[strategy][1]
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](
            self.local_X_train,
            self.local_Y_train,
            self.config.EXP_RANDOM_SEED,
            **additional_params,
        )

    def query_AL_strategy(self) -> SampleIndiceList:
        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")

        warnings.filterwarnings("error")
        from resources.data_types import AL_STRATEGY

        result = None
        bs = self.config.EXP_BATCH_SIZE
        match self.config.EXP_STRATEGY:
            case AL_STRATEGY.PLAYGROUND_INFORMATIVE_DIVERSE:
                result = self.al_strategy.select_batch_(
                    self.model, self.local_train_labeled_idx, bs
                )
            case AL_STRATEGY.PLAYGROUND_MARGIN:
                result = self.al_strategy.select_batch_(
                    self.model, self.local_train_labeled_idx, bs
                )
            case AL_STRATEGY.PLAYGROUND_MIXTURE:
                result = self.al_strategy.select_batch_(
                    self.local_train_labeled_idx, bs
                )
            case AL_STRATEGY.PLAYGROUND_UNIFORM:
                result = self.al_strategy.select_batch_(
                    self.local_train_labeled_idx, bs
                )
            case AL_STRATEGY.PLAYGROUND_GRAPH_DENSITY:
                result = self.al_strategy.select_batch_(
                    bs, self.local_train_labeled_idx
                )
            case AL_STRATEGY.PLAYGROUND_HIERARCHICAL_CLUSTER:
                result = self.al_strategy.select_batch_(
                    bs,
                    self.local_train_labeled_idx,
                    bs,
                    self.local_train_labeled_idx,
                    self.Y,
                )
            case AL_STRATEGY.PLAYGROUND_KCENTER_GREEDY:
                result = self.al_strategy.select_batch_(
                    self.model, self.local_train_labeled_idx, bs
                )
            case AL_STRATEGY.PLAYGROUND_BANDIT:
                result = self.al_strategy.select_batch_(
                    self.local_train_labeled_idx,
                    bs,
                    self.model.score(self.X, self.Y),
                    model=self.model,
                )
            case AL_STRATEGY.PLAYGROUND_MCM:
                result = self.al_strategy.select_batch_(
                    self.model, bs, self.local_train_labeled_idx
                )
        return result

    def prepare_dataset(self):
        pass
