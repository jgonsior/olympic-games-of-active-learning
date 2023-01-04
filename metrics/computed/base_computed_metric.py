from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from misc.config import Config


class Base_Computed_Metric(ABC):
    metrics: List[str]

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def computed_metric_appendix(self) -> str:
        ...

    @abstractmethod
    def apply_to_row(self, row: pd.Series) -> pd.Series:
        pass

    def convert_original_df(self, original_df: pd.DataFrame) -> pd.DataFrame:
        # do stuff using lambda etc
        original_df["computed_metric"] = original_df.apply(
            lambda x: self.apply_to_row(x), axis=1
        )
        return original_df

    def compute(self) -> None:
        for EXP_STRATEGY in self.config.EXP_GRID_STRATEGY:
            for EXP_DATASET in self.config.EXP_GRID_DATASET:
                for metric in self.metrics:
                    # iterate over all experiments/datasets defined for this experiment
                    METRIC_RESULTS_FILE = Path(
                        self.config.OUTPUT_PATH
                        / EXP_STRATEGY.name
                        / EXP_DATASET.name
                        / str(metric + ".csv.gz")
                    )
                    if METRIC_RESULTS_FILE.exists():
                        original_df = pd.read_csv(METRIC_RESULTS_FILE)
                        exp_unique_id_column = original_df["EXP_UNIQUE_ID"]
                        del original_df["EXP_UNIQUE_ID"]

                        new_df = self.convert_original_df(original_df)
                        new_df = new_df.loc[:, ["computed_metric"]]
                        new_df["EXP_UNIQUE_ID"] = exp_unique_id_column

                        # save new df somehow
                        new_df.to_csv(
                            Path(
                                METRIC_RESULTS_FILE.parent
                                / str(
                                    self.computed_metric_appendix()
                                    + "_"
                                    + metric
                                    + ".csv.gz"
                                )
                            ),
                            compression="infer",
                            index=False,
                        )
