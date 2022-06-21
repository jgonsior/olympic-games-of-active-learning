from framework_runners.base_runner import AL_Experiment
from libact.base.dataset import Dataset
from libact.models import LogisticRegression


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
        from ressources.data_types import AL_STRATEGY
        from ressources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        self.al_strategy = al_strategy_to_python_classes_mapping[strategy](
            self.trn_ds, method="lc", model=LogisticRegression()
        )  # TODO

    def query_AL_strategy(self) -> List[int]:
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
