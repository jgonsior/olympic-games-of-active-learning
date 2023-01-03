from __future__ import annotations
import os

from typing import TYPE_CHECKING

import numpy as np
from framework_runners.base_runner import AL_Experiment


if TYPE_CHECKING:
    pass

    from resources.data_types import (
        SampleIndiceList,
    )


class ALIPY_AL_Experiment(AL_Experiment):
    def get_AL_strategy(self):
        from resources.data_types import AL_STRATEGY
        from resources.data_types import (
            al_strategy_to_python_classes_mapping,
        )

        al_strategy = AL_STRATEGY(self.config.EXP_STRATEGY)

        additional_params = al_strategy_to_python_classes_mapping[al_strategy][1]

        if al_strategy in [
            AL_STRATEGY.ALIPY_GRAPH_DENSITY,
            AL_STRATEGY.ALIPY_CORESET_GREEDY,
        ]:
            additional_params["train_idx"] = np.array(
                list(self.map_local_to_global_train_ix.keys())
            )
        elif al_strategy in [
            AL_STRATEGY.ALIPY_LAL,
            AL_STRATEGY.ALIPY_UNCERTAINTY_DTB,
            AL_STRATEGY.ALIPY_BMDR,
            AL_STRATEGY.ALIPY_SPAL,
        ]:
            if len(np.unique(self.Y)) > 2:
                print(
                    "ALIPY_LAL is only implementend for binary classification, exiting…"
                )
                exit(-1)

        al_strategy = al_strategy_to_python_classes_mapping[al_strategy][0](
            X=self.local_X_train,
            y=self.local_Y_train,
            **additional_params,
        )

        if self.config.EXP_STRATEGY == AL_STRATEGY.ALIPY_LAL:
            # check if model has been already trained
            if not os.path.exists(al_strategy._iter_path + "_model"):
                print("Calculating LAL model")
                al_strategy.download_data()
                al_strategy.train_selector_from_file()

                from joblib import dump

                dump(
                    al_strategy._selector,
                    al_strategy._iter_path + "_model",
                    compress=True,
                )
            else:
                print("Loading LAL model from file")
                from joblib import load

                al_strategy._selector = load(al_strategy._iter_path + "_model")
                print("end loaded")

        self.al_strategy = al_strategy

    def query_AL_strategy(self) -> SampleIndiceList:
        select_ind = self.al_strategy.select(
            label_index=self.local_train_labeled_idx,
            unlabel_index=self.local_train_unlabeled_idx,
            model=self.model,
            batch_size=self.config.EXP_BATCH_SIZE,
        )
        if not isinstance(select_ind, list):
            select_ind = select_ind.tolist()
        return select_ind

    # dataset in numpy format and indice lists are fine as it is
    def prepare_dataset(self):
        pass
