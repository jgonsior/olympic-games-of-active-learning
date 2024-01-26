from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import DATASET

from metrics.computed.base_computed_metric import Base_Computed_Metric

from typing import Any, Callable, Dict, TYPE_CHECKING, List, Tuple


if TYPE_CHECKING:
    from resources.data_types import SAMPLES_CATEGORIZER


class DATASET_CATEGORIZATION(Base_Computed_Metric):
    dataset_categorizations: Dict[DATASET, Dict[SAMPLES_CATEGORIZER, np.ndarray]] = {}

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

        if EXP_DATASET not in self.dataset_categorizations.keys():
            self.dataset_categorizations[EXP_DATASET] = {}
        self.dataset_categorizations[EXP_DATASET][samples_categorizer] = data

        return True

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._convert_selected_indices_to_ast(df, merge=False)
        del df["EXP_UNIQUE_ID"]
        return df

    def count_batch_sample_categories(
        self,
        row: pd.Series,
        samples_categorizer: SAMPLES_CATEGORIZER,
        EXP_DATASET: DATASET,
    ) -> pd.Series:
        for ix, r in row.items():
            row[ix] = self.dataset_categorizations[EXP_DATASET][samples_categorizer][
                r
            ].tolist()
        return row

    def get_all_metric_jobs(self) -> List[Tuple[Callable, List[Any]]]:
        from resources.data_types import SAMPLES_CATEGORIZER

        result = []
        for samples_categorizer in SAMPLES_CATEGORIZER:
            result = [
                *result,
                *self._compute_single_metric_jobs(
                    existing_metric_names=["selected_indices"],
                    new_metric_name=f"{samples_categorizer.name}",
                    apply_to_row=self.count_batch_sample_categories,
                    additional_apply_to_row_kwargs={
                        "samples_categorizer": samples_categorizer
                    },
                ),
            ]
        return result
