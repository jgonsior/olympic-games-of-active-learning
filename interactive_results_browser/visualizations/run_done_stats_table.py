from __future__ import annotations
import copy
import enum

import itertools
from matplotlib.pyplot import table
import pandas as pd
from datasets import DATASET
from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple

from typing import Any, Dict

from resources.data_types import AL_STRATEGY

if TYPE_CHECKING:
    from misc.config import Config


class Run_Done_Stats_Table(Base_Visualizer):
    def _load_workload_csv_files(
        self,
        config: Config,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        done_workload: pd.DataFrame = pd.read_csv(config.DONE_WORKLOAD_PATH)  # type: ignore
        open_workload: pd.DataFrame = pd.read_csv(config.WORKLOAD_FILE_PATH)  # type: ignore
        full_workload = pd.concat([done_workload, open_workload])
        full_workload.drop_duplicates(subset="EXP_UNIQUE_ID", inplace=True)

        open_jobs = full_workload.loc[
            full_workload["EXP_UNIQUE_ID"].isin(open_workload["EXP_UNIQUE_ID"])
        ]
        done_jobs = full_workload.loc[
            full_workload["EXP_UNIQUE_ID"].isin(done_workload["EXP_UNIQUE_ID"])
        ]

        return full_workload, open_jobs, done_jobs

    def _update_get_params(self, **kwargs):
        updated_params = {
            k: [int(vv) if isinstance(vv, enum.Enum) else vv for vv in v]
            for k, v in self._exp_grid_request_params.items()
        }

        for k, v in kwargs.items():
            updated_params[k] = v

        return updated_params

    def get_template_data(self) -> Dict[str, Any]:
        full_workload, open_jobs, done_jobs = self._load_workload_csv_files(
            self._config
        )
        if len(self._exp_grid_request_params) == 0:
            # do not filter at all!
            pass
        else:
            # only focus on the specified samples
            for k, v in self._exp_grid_request_params.items():
                k = k.replace("_GRID", "")

                if k in ["VISUALIZATIONS"]:
                    continue

                v = [int(vv) for vv in v]

                full_workload = full_workload.loc[full_workload[k].isin(v)]
                done_jobs = done_jobs.loc[done_jobs[k].isin(v)]

        dataset_strat_counts = {}
        datasets = full_workload["EXP_DATASET"].unique().tolist()
        strategies = full_workload["EXP_STRATEGY"].unique().tolist()

        for dataset, strat in itertools.product(datasets, strategies):
            dataset_strat_counts[(dataset, strat)] = 0

        for dataset, strat in zip(done_jobs.EXP_DATASET, done_jobs.EXP_STRATEGY):
            dataset_strat_counts[(dataset, strat)] += 1

        for dataset, strat in itertools.product(datasets, strategies):
            open_count = int(
                full_workload.loc[
                    (full_workload["EXP_STRATEGY"] == strat)
                    & (full_workload["EXP_DATASET"] == dataset)
                ].count()[0]
            )

            dataset_strat_counts[(dataset, strat)] = (
                dataset_strat_counts[(dataset, strat)],
                open_count,
            )
        table_data = [[""] + [AL_STRATEGY(strat) for strat in strategies]] + [
            ([DATASET(dataset)] + [0 for strat in strategies]) for dataset in datasets
        ]

        for (dataset, strat), count in dataset_strat_counts.items():
            # convert dataset and strat to indices
            dataset = datasets.index(dataset) + 1  # because of column headers
            strat = strategies.index(strat) + 1
            table_data[dataset][strat] = count

        # sort columns, rows -> using pandas
        table_data_df = pd.DataFrame(table_data)
        table_data_df = table_data_df.sort_values(
            by=0, axis=0, key=lambda x: [str(x2) for x2 in x]
        )
        table_data_df = table_data_df.sort_values(
            by=0, axis=1, key=lambda x: [str(x2) for x2 in x]
        )

        rows = list(table_data_df.values.tolist())
        rows[0][0] = "Dataset"

        return {
            "column_names": rows[0],
            "row_data": rows[1:],
            "link_column": "Dataset",
            "experiment_name": self._experiment_name,
            "zip": zip,
            "str": str,
            "isinstance": isinstance,
            "Iterable": Iterable,
            "exp_grid_request_params": self._exp_grid_request_params,
            "type": type,
            "tuple": tuple,
            "enum": enum.Enum,
            "int": int,
            "update_get_params": self._update_get_params,
        }
