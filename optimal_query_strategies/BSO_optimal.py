from __future__ import annotations
import copy
from enum import unique
import random
from typing import TYPE_CHECKING, List, Literal

from enum import IntEnum

import numpy as np
from pyrsistent import b
from sklearn.metrics import accuracy_score, f1_score
from optimal_query_strategies.base_class import Base_AL_Strategy
from optimal_query_strategies.greedy_optimal import FuturePeakEvalMetric, Greedy_Optimal


if TYPE_CHECKING:
    from ressources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )

# details about BSO: https://www.ijcai.org/proceedings/2021/0634.pdf
class Beeam_Search_Optimal(Greedy_Optimal):
    optimal_order: List[int] = []

    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
        train_idx: SampleIndiceList,
        num_queries: int,
        future_peak_eval_metric: FuturePeakEvalMetric = FuturePeakEvalMetric.ACC,
    ) -> None:
        self.num_queries = num_queries
        super().__init__(
            X,
            Y,
            train_idx,
            future_peak_eval_metric=future_peak_eval_metric,
            amount_of_pre_selections=-1,
        )

    def select(
        self,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size: int,
    ) -> SampleIndiceList:
        if len(self.optimal_order) == 0:
            amount_of_samples_needed = batch_size * self.num_queries

            def _beam_search(d_0):
                d_1 = []
                for i in d_0:
                    all_future_metrics = self._compute_future_metrics_for_batches(
                        [i + [_x] for _x in unlabeled_index if _x not in i],
                        labeled_index,
                        model,
                    )
                    d_1 += all_future_metrics
                d_1 = sorted(d_1, key=lambda tup: tup[0], reverse=True)

                if len(d_1[0][1]) >= amount_of_samples_needed or len(d_1[0][1]) == len(
                    unlabeled_index
                ):
                    return d_1[0][1]
                else:
                    d_1 = [_x[1] for _x in d_1[0:5]]
                    return _beam_search(d_1)

            self.optimal_order = _beam_search([[]])

        selected_indices = self.optimal_order[0:batch_size]

        self.optimal_order = self.optimal_order[batch_size:]
        return selected_indices
