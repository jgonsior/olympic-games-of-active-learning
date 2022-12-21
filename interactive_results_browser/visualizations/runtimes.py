from __future__ import annotations
from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import TYPE_CHECKING, Any, List

from typing import Any, Dict


if TYPE_CHECKING:
    pass
import seaborn as sns
import matplotlib.pyplot as plt
from interactive_results_browser.cache import memory


def _plot_function(plot_df, metric, my_palette, my_markers):
    fig, ax = plt.subplots(figsize=(8, 4))
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
def _cache_runtimes(metric, done_workload) -> List[str]:
    plot_df = done_workload.filter([metric, "EXP_STRATEGY", "EXP_DATASET"]).sort_values(
        by=metric
    )

    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={"metric": metric},
        plot_function=_plot_function,
        legend_names=done_workload["EXP_STRATEGY"].unique(),
        df_col_key="EXP_DATASET",
        create_legend=False,
        combined_df_col_key_plot=True,
    )
    return plot_urls


class Runtimes(Base_Visualizer):
    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
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

        plot_url = _cache_runtimes(
            metric=metric,
            done_workload=self._load_done_workload(),
        )
        return {"plot_data": plot_url}
