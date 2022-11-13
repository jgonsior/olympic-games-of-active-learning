from __future__ import annotations

# optimized future peak -> only peak into the future for the n most "promising" (or random) samples?
import copy
from enum import unique
import random
from typing import TYPE_CHECKING, List, Literal, Tuple

from enum import IntEnum

import numpy as np
from pyrsistent import b
from sklearn.metrics import accuracy_score, f1_score
from optimal_query_strategies.base_class import Base_AL_Strategy


if TYPE_CHECKING:
    from resources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )


@unique
class FuturePeakEvalMetric(IntEnum):
    ACC = 1
    F1 = 2


class Greedy_Optimal(Base_AL_Strategy):
    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
        amount_of_pre_selections: int = -1,
        future_peak_eval_metric: FuturePeakEvalMetric = FuturePeakEvalMetric.ACC,
    ) -> None:
        super().__init__(X, Y)
        self.amount_of_pre_selections = int(amount_of_pre_selections)
        self.future_peak_eval_metric = FuturePeakEvalMetric(future_peak_eval_metric)  # type: ignore

    def _future_peak(
        self,
        model: LEARNER_MODEL,
        labeled_index: SampleIndiceList,
        unlabeled_sample_indices: SampleIndiceList,
    ) -> float:
        # return random.random()
        copy_of_classifier: LEARNER_MODEL = copy.deepcopy(model)
        copy_of_labeled_mask = [*labeled_index, *unlabeled_sample_indices]

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

    def _compute_future_metrics_for_batches(
        self,
        batches: List[SampleIndiceList],
        labeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
    ) -> List[Tuple[float, SampleIndiceList]]:
        future_peak_acc = []

        for unlabeled_sample_indices in batches:
            future_peak_acc.append(
                self._future_peak(
                    model,
                    labeled_index,
                    unlabeled_sample_indices,
                )
            )
        zero_to_one_values_and_index = list(zip(future_peak_acc, batches))

        ordered_list_of_possible_sample_indices = sorted(
            zero_to_one_values_and_index, key=lambda tup: tup[0], reverse=True
        )

        return ordered_list_of_possible_sample_indices

    def _compute_sorted_future_batches(
        self,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size: int,
    ) -> List[Tuple[float, SampleIndiceList]]:
        if self.amount_of_pre_selections > len(unlabeled_index):
            random_func = random.choices  # with replacement
        else:
            random_func = random.sample  # without replacement

        # fancy conversion to tuples and set to avoid costly duplicate elements
        pre_sampled_X_querie_indices: List[SampleIndiceList] = [
            list(_x)
            for _x in set(
                tuple(random_func(unlabeled_index, k=batch_size))
                for _ in range(0, self.amount_of_pre_selections)
            )
        ]

        return self._compute_future_metrics_for_batches(
            pre_sampled_X_querie_indices, labeled_index, model
        )

    def select(
        self,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size: int,
    ) -> SampleIndiceList:
        ordered_list_of_possible_sample_indices = self._compute_sorted_future_batches(
            labeled_index, unlabeled_index, model, batch_size
        )
        return ordered_list_of_possible_sample_indices[0][
            1
        ]  # [v for _, v in ordered_list_of_possible_sample_indices[:batch_size]] -> this here is for BSO ???
