from __future__ import annotations

# super expensive -> full minimax future peak!
# -> subclass of greedy_optimal.py
from optimal_query_strategies.greedy_optimal import FuturePeakEvalMetric, Greedy_Optimal

# optimized future peak -> only peak into the future for the n most "promising" (or random) samples?
import copy
from enum import unique
import random
from typing import TYPE_CHECKING, List, Literal

from enum import IntEnum

import numpy as np
from pyrsistent import b
from sklearn.metrics import accuracy_score, f1_score
from optimal_query_strategies.base_class import Base_AL_Strategy


if TYPE_CHECKING:
    from ressources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )


class True_Optimal(Greedy_Optimal):
    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
        future_peak_eval_metric: FuturePeakEvalMetric = FuturePeakEvalMetric.ACC,
    ) -> None:
        super().__init__(X, Y, -1, future_peak_eval_metric)

    def select(
        self,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size: int,
    ) -> SampleIndiceList:
        ordered_list_of_possible_sample_indices = (
            self._compute_future_metrics_for_batches(
                [[_x] for _x in unlabeled_index], labeled_index, model
            )
        )

        batch = [
            _x[1][0] for _x in ordered_list_of_possible_sample_indices[0:batch_size]
        ]
        return batch
