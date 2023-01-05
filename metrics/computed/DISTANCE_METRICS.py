from __future__ import annotations
import ast
import itertools
import numpy as np
import pandas as pd
from datasets import DATASET, load_dataset

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY
    from misc.config import Config


class DISTANCE_METRICS(Base_Computed_Metric):
    metrics = ["avg_dist_batch", "avg_dist_labeled", "avg_dist_unlabeled"]

    _precomputed_distances: np.ndarray

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        print("loading", EXP_DATASET)
        self._precomputed_distances = np.load(
            f"{self.config.DATASETS_PATH}/{EXP_DATASET.name}{self.config.DATASETS_DISTANCES_APPENDIX}"
        )["arr_0"]
        print("done loading")

    def computed_metric_appendix(self) -> str:
        return "dist"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(lambda x: [ast.literal_eval(iii) for iii in x], axis=0)
        return df

    def avg_dist_batch(
        self, row: pd.Series, EXP_STRATEGY: AL_STRATEGY, EXP_DATASET: DATASET
    ) -> pd.Series:
        results = 0
        for _, x in row.items():
            distances = []
            for s1, s2 in itertools.combinations(x, 2):
                distances.append(self._precomputed_distances[s1][s2])

            if len(distances) == 0:
                results += 0
            else:
                results += sum(distances) / len(distances)
        return results

    def avg_dist_labeled(
        self, row: pd.Series, EXP_STRATEGY: AL_STRATEGY, EXP_DATASET: DATASET
    ) -> pd.Series:
        results = 0
        for _, x in row.items():
            distances = []
            for s1, s2 in itertools.combinations(x, 2):
                distances.append(self._precomputed_distances[s1][s2])

            if len(distances) == 0:
                results += 0
            else:
                results += sum(distances) / len(distances)
        return results

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_name="selected_indices",
            new_metric_name="avg_dist_batch",
            apply_to_row=self.avg_dist_batch,
        )
        self._take_single_metric_and_compute_new_one(
            existing_metric_name="selected_indices",
            new_metric_name="avg_dist_labeled",
            apply_to_row=self.avg_dist_labeled,
        )
