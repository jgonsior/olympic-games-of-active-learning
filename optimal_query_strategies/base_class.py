"""Optimal/oracle query strategies for OGAL.

This module provides implementations of oracle-based AL strategies that
use knowledge of ground truth labels to make optimal or near-optimal
sample selections. These are used as upper-bound baselines.

Key components:
    - Base_AL_Strategy: Abstract base class for oracle strategies
    - Greedy_Optimal: Greedy selection based on future performance
    - True_Optimal: Exhaustive search for optimal selection
    - Beeam_Search_Optimal: Beam search approximation

These strategies are not practical for real AL but serve as baselines
to evaluate how close practical strategies come to optimal performance.

For more details, see docs/research_reuse.md
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resources.data_types import (
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
