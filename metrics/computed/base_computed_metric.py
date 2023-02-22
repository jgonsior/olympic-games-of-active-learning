from __future__ import annotations
from abc import ABC
import multiprocessing
from pathlib import Path
from typing import List, TYPE_CHECKING
from joblib import Parallel, delayed, parallel_backend
import ast
import pandas as pd

from datasets import DATASET

if TYPE_CHECKING:
    from misc.config import Config


def _process_a_single_strategy(
    EXP_STRATEGY,
    EXP_DATASET,
    existing_metric_names,
    OUTPUT_PATH,
    _pre_appy_to_row_hook,
    convert_original_df,
    apply_to_row,
    new_metric_name,
):
    # iterate over all experiments/datasets defined for this experiment
    metric_result_files: List[Path] = []
    for existing_metric_name in existing_metric_names:
        METRIC_RESULTS_FILE = Path(
            OUTPUT_PATH
            / EXP_STRATEGY.name
            / EXP_DATASET.name
            / str(existing_metric_name + ".csv.xz")
        )
        if not METRIC_RESULTS_FILE.exists():
            continue
        # print(METRIC_RESULTS_FILE)
        metric_result_files.append(METRIC_RESULTS_FILE)

    joined_df = pd.DataFrame()
    for METRIC_RESULTS_FILE in metric_result_files:
        original_df = pd.read_csv(METRIC_RESULTS_FILE, header=0, delimiter=",")

        if len(joined_df) == 0:
            joined_df = original_df
        else:
            joined_df = joined_df.merge(original_df, how="outer", on="EXP_UNIQUE_ID")
            joined_df.drop_duplicates(inplace=True)

    if len(joined_df) == 0:
        #  print(f"{metric_result_files} are all together empty")
        return

    exp_unique_id_column = joined_df["EXP_UNIQUE_ID"]

    joined_df = _pre_appy_to_row_hook(joined_df)

    new_df = convert_original_df(
        joined_df,
        apply_to_row=apply_to_row,
    )

    new_df = new_df.loc[:, ["computed_metric"]]
    new_df["EXP_UNIQUE_ID"] = exp_unique_id_column

    print(metric_result_files[0].parent / str(new_metric_name + ".csv.xz"))
    # save new df somehow
    new_df.to_csv(
        Path(metric_result_files[0].parent / str(new_metric_name + ".csv.xz")),
        compression="infer",
        index=False,
    )


class Base_Computed_Metric(ABC):
    metrics: List[str]
    done_workload_df: pd.DataFrame

    def __init__(self, config: Config) -> None:
        self.done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
        self.config = config

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

    def _convert_selected_indices_to_ast(self, df: pad.DataFrame) -> pd.DataFrame:
        column_names_which_are_al_cycles = list(df.columns)
        column_names_which_are_al_cycles.remove("EXP_UNIQUE_ID")

        df = df.fillna("")
        df["selected_indices"] = df[column_names_which_are_al_cycles].apply(
            lambda x: ast.literal_eval(
                ("[" + ",".join(x).replace("[", "").replace("]", "") + "]").replace(
                    ",,", ""
                )
            ),
            axis=1,
        )
        for c in column_names_which_are_al_cycles:
            del df[c]

        return df

    def _pre_appy_to_row_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _per_dataset_hook(self, EXP_DATASET: DATASET) -> None:
        ...

    def _take_single_metric_and_compute_new_one(
        self,
        existing_metric_names: List[str],
        new_metric_name: str,
        apply_to_row,
    ) -> None:
        for EXP_DATASET in self.config.EXP_GRID_DATASET:
            self._per_dataset_hook(EXP_DATASET)

            with parallel_backend(
                "multiprocessing", n_jobs=1  # multiprocessing.cpu_count()
            ):
                Parallel()(
                    delayed(_process_a_single_strategy)(
                        EXP_STRATEGY,
                        EXP_DATASET,
                        existing_metric_names,
                        self.config.OUTPUT_PATH,
                        self._pre_appy_to_row_hook,
                        self.convert_original_df,
                        apply_to_row,
                        new_metric_name,
                    )
                    for EXP_STRATEGY in self.config.EXP_GRID_STRATEGY
                )

    def compute(self) -> None:
        for metric in self.metrics:
            self._take_single_metric_and_compute_new_one(
                existing_metric_names=[metric],
                new_metric_name=self.computed_metric_appendix() + "_" + metric,
                apply_to_row=self.apply_to_row,
            )
