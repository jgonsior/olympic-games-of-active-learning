from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING

import modin.pandas as pd

from datasets import DATASET

if TYPE_CHECKING:
    from resources.data_types import AL_STRATEGY
    from misc.config import Config


class Base_Computed_Metric(ABC):
    metrics: List[str]
    done_workload_df: pd.DataFrame

    def __init__(self, config: Config) -> None:
        self.done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
        self.config = config

    @abstractmethod
    def computed_metric_appendix(self) -> str:
        ...

    def apply_to_row(self, row: pd.Series) -> pd.Series:
        pass

    def convert_original_df(
        self,
        original_df: pd.DataFrame,
        apply_to_row,
    ) -> pd.DataFrame:
        # do stuff using lambda etc
        original_df["computed_metric"] = original_df.apply(
            lambda x: apply_to_row(x), axis=1
        )
        return original_df

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        ...

    def _take_single_metric_and_compute_new_one(
        self, existing_metric_name: str, new_metric_name: str, apply_to_row
    ) -> None:
        for EXP_DATASET in self.config.EXP_GRID_DATASET:
            self._per_dataset_hook(EXP_DATASET)
            for EXP_STRATEGY in self.config.EXP_GRID_STRATEGY:
                # iterate over all experiments/datasets defined for this experiment
                METRIC_RESULTS_FILE = Path(
                    self.config.OUTPUT_PATH
                    / EXP_STRATEGY.name
                    / EXP_DATASET.name
                    / str(existing_metric_name + ".csv.gz")
                )
                if METRIC_RESULTS_FILE.exists():
                    original_df = pd.read_csv(METRIC_RESULTS_FILE, header=0)
                    exp_unique_id_column = original_df["EXP_UNIQUE_ID"]

                    original_df = self._pre_appy_to_row_hook(original_df)

                    new_df = self.convert_original_df(
                        original_df,
                        apply_to_row=apply_to_row,
                    )
                    new_df = new_df.loc[:, ["computed_metric"]]
                    new_df["EXP_UNIQUE_ID"] = exp_unique_id_column

                    # save new df somehow
                    new_df.to_csv(
                        Path(
                            METRIC_RESULTS_FILE.parent
                            / str(new_metric_name + ".csv.gz")
                        ),
                        compression="infer",
                        index=False,
                    )

    def compute(self) -> None:
        for metric in self.metrics:
            self._take_single_metric_and_compute_new_one(
                existing_metric_name=metric,
                new_metric_name=self.computed_metric_appendix() + "_" + metric,
                apply_to_row=self.apply_to_row,
            )
