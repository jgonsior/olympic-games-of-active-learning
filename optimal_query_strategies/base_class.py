from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ressources.data_types import (
        LEARNER_MODEL,
        FeatureVectors,
        LabelList,
        SampleIndiceList,
    )


class Base_AL_Strategy(ABC):
    def __init__(
        self,
        X: FeatureVectors,
        Y: LabelList,
    ) -> None:
        self.X = X
        self.Y = Y

    @abstractmethod
    def select(
        self,
        X: FeatureVectors,
        Y: LabelList,
        labeled_index: SampleIndiceList,
        unlabeled_index: SampleIndiceList,
        model: LEARNER_MODEL,
        batch_size: int,
    ) -> SampleIndiceList:
        ...
