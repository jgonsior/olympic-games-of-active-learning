# optimized future peak -> only peak into the future for the n most "promising" (or random) samples?
import copy
from enum import unique
from typing import Literal

from enum import IntEnum

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from base_class import Base_AL_Strategy
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
    F1 = 1


class Greedy_Optimal(Base_AL_Strategy):
    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
        heuristic: GreedyHeuristic = GreedyHeuristic.NONE,
        max_peaks: int = -1,
        future_peak_eval_metric: FuturePeakEvalMetric = FuturePeakEvalMetric.ACC,
    ) -> None:
        self.heuristic = heuristic
        self.max_peaks = max_peaks
        self.future_peak_eval_metric = future_peak_eval_metric
        self.__

    def _future_peak(
        self,
        model: LEARNER_MODEL,
        labeled_index: SampleIndiceList,
        unlabeled_sample_indices: SampleIndiceList,
    ) -> float:
        copy_of_classifier = copy.deepcopy(model)

        copy_of_labeled_mask = np.append(
            labeled_index, unlabeled_sample_indices, axis=0
        )

        copy_of_classifier.fit(
            self.X[copy_of_labeled_mask],
            self.Y[copy_of_labeled_mask],
        )

        Y_pred_test = copy_of_classifier.predict(self.X)
        Y_true = self.Y

        if self.future_peak_eval_metric == FuturePeakEvalMetric.ACC:
            future_metric_value_with_that_label = accuracy_score(Y_pred_test, Y_true)
        elif self.future_peak_eval_metric == FuturePeakEvalMetric.F1:
            future_metric_value_with_that_label = f1_score(Y_pred_test, Y_true)
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
        if self.max_peaks > len(unlabeled_index):
            replace = True
        else:
            replace = False

        if self.heuristic == GreedyHeuristic.RANDOM:
            pre_sampled_X_querie_indices: SampleIndiceList = list(
                np.random.choice(
                    np.array(unlabeled_index),
                    size=self.max_peaks,
                    replace=replace,
                ).tolist()
            )
        else:
            raise ValueError(f"{self.heuristic} is not yet implemented")

        future_peak_acc = []
        # single thread
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
