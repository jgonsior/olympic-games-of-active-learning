from __future__ import annotations
import enum

import itertools
import pandas as pd
from datasets import DATASET
from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, Iterable, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Run_Done_Stats_Table(Base_Visualizer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def template_name(self) -> str:
        return "run_done_stats.html.j2"

    def get_data(self) -> Dict[str, Any]:
        return super().get_data()

    def load_workload_csv_files(
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

    def _plot_run_done_stats(config: Config, exp_grid_request_params):
        full_workload, open_jobs, done_jobs = self._load_workload_csv_files(config)
        if len(exp_grid_request_params) == 0:
            # do not filter at all!
            pass
        else:
            # only focus on the specified samples
            for k, v in exp_grid_request_params.items():
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

        table_data = [[""] + [strat for strat in strategies]] + [
            ([DATASET(dataset)] + [0 for strat in strategies]) for dataset in datasets
        ]

        for (dataset, strat), count in dataset_strat_counts.items():
            # convert dataset and strat to indices
            dataset = datasets.index(dataset) + 1  # because of column headers
            strat = strategies.index(strat) + 1
            table_data[dataset][strat] = count

        # sort columns, rows -> using pandas
        table_data_df = pd.DataFrame(table_data)

        # sort after dataset NAME
        dataset_names = table_data_df[1:][0].values.tolist()
        dataset_names_str = [str(x.name).lower() for x in dataset_names]
        sorted_range = sorted(
            range(len(dataset_names_str)), key=dataset_names_str.__getitem__
        )
        dataset_names_ints = list(range(len(dataset_names_str)))
        new_sorting = [sorted_range.index(x) for x in dataset_names_ints]
        custom_sorting = {
            idx: value + 1 for idx, value in zip(dataset_names, new_sorting)
        }
        custom_sorting[""] = 0

        table_data_df.sort_values(
            by=[table_data_df.columns[0]],
            key=lambda x: x.map(lambda col: custom_sorting[col]),
            inplace=True,
        )

        # same for header columns
        column_names = table_data_df.columns.values.tolist()[1:]
        strategy_names = [str(x).lower() for x in table_data_df.loc[0][1:]]
        yx = list(zip(strategy_names, column_names))
        yx_sort = sorted(yx)
        sorted_column_titles = [0] + [x[1] for x in yx_sort]

        rows = list(table_data_df.values.tolist())
        rows[0][0] = "Dataset"

        return {
            "column_names": rows[0],
            "row_data": rows[1:],
            "link_column": "Dataset",
            "zip": zip,
            "str": str,
            "isinstance": isinstance,
            "Iterable": Iterable,
            "exp_grid_request_params": exp_grid_request_params,
            "type": type,
            "tuple": tuple,
            "enum": enum.Enum,
            "int": int,
        }
