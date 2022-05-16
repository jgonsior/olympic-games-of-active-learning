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


if TYPE_CHECKING:
    from ressources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )

# details about BSO: https://www.ijcai.org/proceedings/2021/0634.pdf
class Beeam_Search_Optimal(Base_AL_Strategy):
    def __init__(self, X: FeatureVectors, Y: LabelList) -> None:
        super().__init__(X, Y)

    def select(
        self,
        X: FeatureVectors,
        Y: LabelList,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size=...,
    ) -> SampleIndiceList:
        return super().select(X, Y, labeled_index, unlabeled_index, model, batch_size)
