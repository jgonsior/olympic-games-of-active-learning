from __future__ import annotations

import numpy as np
from framework_runners.base_runner import AL_Experiment
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SklearnProbaAdapter
from libact.query_strategies import UncertaintySampling

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
    )


class LIBACT_Experiment(AL_Experiment):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.trn_ds = None
        self.fully_labeled_trn_ds = None
        self.al_strategy = None

    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        if strategy in [AL_STRATEGY.LIBACT_VR]:
            from resources.data_types import _import_compiled_libact_strategies

            _import_compiled_libact_strategies()

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
            additional_params["classes"] = np.unique(self.local_Y_train).tolist()

        self.al_strategy = al_strategy_to_python_classes_mapping[strategy][0](
            self.trn_ds, **additional_params
        )

    def query_AL_strategy(self) -> SampleIndiceList:
        from resources.data_types import AL_STRATEGY

        if self.al_strategy is None:
            from misc.Errors import NoStrategyError

            raise NoStrategyError("get_AL_strategy() has to be called before querying")
        ret = []
        batch_size = self.config.EXP_BATCH_SIZE

        match self.config.EXP_STRATEGY:
            case AL_STRATEGY.LIBACT_VR:
                ret = self.al_strategy.make_n_queries(
                    batch_size=self.config.EXP_BATCH_SIZE
                )
            case AL_STRATEGY.LIBACT_HINTSVM:
                ret = self.al_strategy.make_n_queries(
                    batch_size=self.config.EXP_BATCH_SIZE
                )
            case AL_STRATEGY.LIBACT_DWUS:
                ret = self.al_strategy.make_n_queries(
                    batch_size=self.config.EXP_BATCH_SIZE
                )
            case AL_STRATEGY.LIBACT_ALBL:
                raise ValueError("ALBL not yet implemented")
                pass  # TODO
            case AL_STRATEGY.LIBACT_QUIRE:
                ret = self.al_strategy.make_n_queries(
                    batch_size=self.config.EXP_BATCH_SIZE
                )
            case AL_STRATEGY.LIBACT_UNCERTAINTY_SM | AL_STRATEGY.LIBACT_UNCERTAINTY_ENT | AL_STRATEGY.LIBACT_UNCERTAINTY_LC:
                unlabeled_entry_ids, scores = zip(*self.al_strategy._get_scores())

                max_ids = np.argpartition(
                    -np.array(scores), self.config.EXP_BATCH_SIZE
                )[: self.config.EXP_BATCH_SIZE]
                ret = np.array(unlabeled_entry_ids)[max_ids]
            case AL_STRATEGY.LIBACT_EER:
                ret = self.al_strategy.make_n_queries(
                    batch_size=self.config.EXP_BATCH_SIZE
                )
            case AL_STRATEGY.LIBACT_HIERARCHICAL_SAMPLING:
                raise ValueError("ALBL not yet implemented")
                pass  # TODO
            case AL_STRATEGY.LIBACT_QBC:
                raise ValueError("ALBL not yet implemented")
                pass  # TODO
            case _:
                from misc.Errors import WrongFrameworkError

                raise WrongFrameworkError(
                    "Libact runner was called with a non-Libact strategy"
                )
        lb = self.local_Y_train[ret]

        for k, v in zip(ret, lb):
            self.trn_ds.update(k, v)

        if not isinstance(ret, list):
            ret = ret.tolist()
        return ret

    def prepare_dataset(self):
        init_labeled_mask = np.array([None for _ in self.local_Y_train])
        for labeled_idx in self.local_train_labeled_idx:
            init_labeled_mask[labeled_idx] = self.local_Y_train[labeled_idx]
        self.trn_ds = Dataset(self.local_X_train, init_labeled_mask)
