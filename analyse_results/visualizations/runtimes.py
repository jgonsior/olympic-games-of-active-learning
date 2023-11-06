from __future__ import annotations
from pathlib import Path
from re import M

import pandas as pd
from analyse_results.visualizations.base_visualizer import (
    MERGE_AL_CYCLE_METRIC_STRATEGY,
    Base_Visualizer,
)
from typing import TYPE_CHECKING, Any, List

from typing import Any, Dict


if TYPE_CHECKING:
    pass
import seaborn as sns
import matplotlib.pyplot as plt
from analyse_results.cache import memory


def _plot_function(plot_df, metric, my_palette, my_markers):
    _, ax = plt.subplots(figsize=(8, 4))
    ax = sns.barplot(
        data=plot_df,
        y="EXP_STRATEGY",
        x=metric,
        palette=my_palette,
        ax=ax,
    )
    ax.set(ylabel=None)
    ax.set_xscale("log")
    for container in ax.containers:
        ax.bar_label(container, padding=10, fmt="%.2g")
    return ax


@memory.cache()
def _cache_runtimes(
    done_workload_df: pd.DataFrame, metric, OUTPUT_PATH: Path
) -> List[str]:
    done_workload_df = done_workload_df.loc[
        :, ["EXP_UNIQUE_ID", "EXP_STRATEGY", "EXP_DATASET"]
    ]

    plot_df = Base_Visualizer.load_detailed_metric_files(
        done_workload_df,
        metric,
        OUTPUT_PATH,
        merge_al_cycle_metrics=MERGE_AL_CYCLE_METRIC_STRATEGY.ORIGINAL,
    )

    del plot_df["EXP_UNIQUE_ID"]

    plot_df = plot_df.melt(
        id_vars=["EXP_STRATEGY", "EXP_DATASET"], var_name="AL Cycle", value_name=metric
    )

    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={"metric": metric},
        plot_function=_plot_function,
        legend_names=done_workload_df["EXP_STRATEGY"].unique(),
        df_col_key="EXP_DATASET",
        create_legend=False,
        combined_df_col_key_plot=True,
    )
    return plot_urls


class Runtimes(Base_Visualizer):
    @staticmethod
    def get_additional_request_params(
        OUTPUT_PATH: Path, with_basic=True
    ) -> Dict[str, List[Any]]:
        return {
            "VIZ_RT_METRIC": [
                "learner_training_time",
                "query_selection_time",
            ]
        }

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_RT_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_RT_METRIC value"}

        metric = self._exp_grid_request_params["VIZ_RT_METRIC"][0]

        done_workload_df = self._load_done_workload()
        plot_url = _cache_runtimes(
            metric=metric,
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
        )
        return {"plot_data": plot_url}
