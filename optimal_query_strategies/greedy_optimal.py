from __future__ import annotations

# optimized future peak -> only peak into the future for the n most "promising" (or random) samples?
import copy
from enum import unique
from typing import TYPE_CHECKING, List, Literal

from enum import IntEnum

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from optimal_query_strategies.base_class import Base_AL_Strategy


if TYPE_CHECKING:
    from ressources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )


@unique
class GreedyHeuristic(IntEnum):
    NONE = 0
    RANDOM = 1
    COSINE = 2
    UNCERTAINTY = 3


@unique
class FuturePeakEvalMetric(IntEnum):
    ACC = 1
    F1 = 2


class Greedy_Optimal(Base_AL_Strategy):
    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
        heuristic: GreedyHeuristic = GreedyHeuristic.NONE,
        amount_of_pre_selections: int = -1,
        future_peak_eval_metric: FuturePeakEvalMetric = FuturePeakEvalMetric.ACC,
    ) -> None:
        self.heuristic = GreedyHeuristic[heuristic]  # type: ignore
        self.amount_of_pre_selections = int(amount_of_pre_selections)
        self.future_peak_eval_metric = FuturePeakEvalMetric[future_peak_eval_metric]  # type: ignore
        self.X = X
        self.Y = Y

    def _future_peak(
        self,
        model: LEARNER_MODEL,
        labeled_index: SampleIndiceList,
        unlabeled_sample_indices: SampleIndiceList,
    ) -> float:
        copy_of_classifier: LEARNER_MODEL = copy.deepcopy(model)

        copy_of_labeled_mask = np.append(
            labeled_index, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(  # type: ignore
            self.X[copy_of_labeled_mask],
            self.Y[copy_of_labeled_mask],
        )

        Y_pred_test = copy_of_classifier.predict(self.X)  # type: ignore
        Y_true = self.Y

        if self.future_peak_eval_metric == FuturePeakEvalMetric.ACC:
            future_metric_value_with_that_label = accuracy_score(Y_pred_test, Y_true)
        elif self.future_peak_eval_metric == FuturePeakEvalMetric.F1:
            future_metric_value_with_that_label = f1_score(
                Y_pred_test, Y_true, average="macro", zero_division=0  # type: ignore
            )
        else:
            raise ValueError("No enum defined, error!")

        return future_metric_value_with_that_label

    def select(
        self,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size=int,
    ) -> SampleIndiceList:
        if self.amount_of_pre_selections > len(unlabeled_index):
            replace = True
        else:
            replace = False

        if self.heuristic == GreedyHeuristic.RANDOM:
            pre_sampled_X_querie_indices: List[SampleIndiceList] = np.random.choice(
                np.array(unlabeled_index),
                size=self.amount_of_pre_selections,
                replace=replace,
            ).tolist()  # type: ignore

        else:
            raise ValueError(f"{self.heuristic} is not yet implemented")

        future_peak_acc = []

        for unlabeled_sample_indices in pre_sampled_X_querie_indices:
            future_peak_acc.append(
                self._future_peak(
                    model,
                    labeled_index,
                    unlabeled_sample_indices,
                )
            )

        zero_to_one_values_and_index = set(
            zip(future_peak_acc, pre_sampled_X_querie_indices)
        )

        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return [v for _, v in ordered_list_of_possible_sample_indices[:batch_size]]
