from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Dict, List, TYPE_CHECKING, Literal
from scipy.spatial import distance


if TYPE_CHECKING:
    from resources.data_types import SAMPLES_CATEGORIZER


class DATASET_CATEGORIZATION(Base_Computed_Metric):
    dataset_categorizations: Dict[SAMPLES_CATEGORIZER, np.ndarray] = {}

    def _per_dataset_hook(
        self, EXP_DATASET: DATASET, samples_categorizer: SAMPLES_CATEGORIZER
    ) -> bool:
        print("parsing ", EXP_DATASET)
        samples_categorization_path = Path(
            f"{self.config.OUTPUT_PATH}/_{samples_categorizer.name}/{EXP_DATASET.name}.npz"
        )

        if not samples_categorization_path.exists():
            print("Please compute ", samples_categorization_path, " first")
            return False

        data = np.load(samples_categorization_path)["samples_categorization"]

        self.dataset_categorizations[samples_categorizer] = data

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._convert_selected_indices_to_ast(df)

    def count_batch_sample_categories(
        self, row: pd.Series, samples_categorizer: SAMPLES_CATEGORIZER
    ) -> pd.Series:
        values = self.dataset_categorizations[samples_categorizer][
            row["selected_indices"]
        ]
        return values

    def compute(self) -> None:
        from resources.data_types import SAMPLES_CATEGORIZER

        for samples_categorizer in SAMPLES_CATEGORIZER:
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=["selected_indices"],
                new_metric_name=f"{samples_categorizer.name}",
                apply_to_row=self.count_batch_sample_categories,
                additional_apply_to_row_kwargs={
                    "samples_categorizer": samples_categorizer
                },
            )
