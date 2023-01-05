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

    _precomputed_distances: Dict[DATASET, np.ndarray] = {}

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # compute distance matrix for all datasets
        for dataset in self.config.EXP_GRID_DATASET:
            dataset_df, splits = load_dataset(dataset, self.config)
            X = dataset_df.loc[:, dataset_df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
            distances = pairwise_distances(X, X, metric="cosine", n_jobs=-1)
            self._precomputed_distances[dataset] = distances

    def avg_dist_batch(self) -> None:
        ...

    def computed_metric_appendix(self) -> str:
        return "dist"

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.apply(lambda x: [ast.literal_eval(iii) for iii in x], axis=0)
        return df

    def apply_to_row(
        self, row: pd.Series, EXP_STRATEGY: AL_STRATEGY, EXP_DATASET: DATASET
    ) -> pd.Series:
        results = 0
        for _, x in row.items():
            distances = []
            for s1, s2 in itertools.combinations(x, 2):
                distances.append(self._precomputed_distances[EXP_DATASET][s1][s2])

            if len(distances) == 0:
                results += 0
            else:
                results += sum(distances) / len(distances)
        return results

    def compute(self) -> None:
        self._take_single_metric_and_compute_new_one(
            existing_metric_name="selected_indices", new_metric_name="avg_dist_batch"
        )
