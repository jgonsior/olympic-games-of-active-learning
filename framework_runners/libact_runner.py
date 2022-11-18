from distutils.command.config import config

import numpy as np
from framework_runners.base_runner import AL_Experiment
from typing import List
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SklearnProbaAdapter, SVM
from libact.query_strategies import UncertaintySampling

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from misc.config import Config


class LIBACT_Experiment(AL_Experiment):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.fully_labeled = None
        self.trn_ds = None
        self.al_strategy = None

    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)

        additional_params = al_strategy_to_python_classes_mapping[strategy][1]
        if (
            self.config.EXP_LEARNER_MODEL == "LOG_REG"
            or self.config.EXP_LEARNER_MODEL == "SVM_LIBACT"
        ):
            additional_params["model"] = self.model
        else:
            additional_params["model"] = SklearnProbaAdapter(self.model)

        additional_params["random_state"] = self.config.EXP_RANDOM_SEED

        if self.config.EXP_STRATEGY == AL_STRATEGY.LIBACT_ALBL:
            from libact.query_strategies import HintSVM

            additional_params["T"] = 100
            additional_params["query_strategies"] = [
                UncertaintySampling(self.trn_ds, model=LogisticRegression(C=1.0)),
                UncertaintySampling(self.trn_ds, model=LogisticRegression(C=0.01)),
                HintSVM(self.trn_ds),
            ]
        elif self.config.EXP_STRATEGY == AL_STRATEGY.LIBACT_HIERARCHICAL_SAMPLING:
            del additional_params["model"]
            additional_params["classes"] = np.unique(self.Y[self.train_idx]).tolist()

        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](
            self.trn_ds, **additional_params
        )

    def query_AL_strategy(self) -> List[int]:
        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        ret = []
        select_id, scores = self.al_strategy.make_query(return_score=True)
        ret.append(select_id)
        batch_size = self.config.EXP_BATCH_SIZE
        ids, scores = zip(*scores)
        i = 0
        while i < batch_size:
            maximum = max(scores)
            max_ind = scores.index(maximum)
            ret.append(ids[max_ind])
        return ret

    def prepare_dataset(self):
        self.fully_labeled = Dataset(self.X, self.Y)
        self.trn_ds = Dataset(
            (self.X[self.unlabel_idx].tolist() + self.X[self.label_idx].tolist()),
            ([None] * len(self.unlabel_idx) + self.Y[self.label_idx].tolist()),
        )
