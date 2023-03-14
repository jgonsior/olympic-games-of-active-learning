from __future__ import annotations
from pathlib import Path

import pandas as pd

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, List, TYPE_CHECKING

from typing import Any, Dict

from metrics.Standard_ML_Metrics import Standard_ML_Metrics


if TYPE_CHECKING:
    pass
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from interactive_results_browser.cache import memory


def _plot_function(plot_df, metric, my_palette, my_markers):
    fig, ax = plt.subplots(figsize=(8, 4))

    rel = sns.lineplot(
        plot_df,
        ax=ax,
        x="AL Cycle",
        y=metric,
        style="EXP_STRATEGY",
        # markers="Strategy",
        markers=my_markers,
        palette=my_palette,
        hue="EXP_STRATEGY",
    )
    rel.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    rel.set(ylim=(0, 1))
    return rel


@memory.cache()
def _cache_create_plots(
    done_workload_df: pd.DataFrame,
    metric: str,
    OUTPUT_PATH: Path,
) -> List[str]:
    done_workload_df = done_workload_df.loc[
        :, ["EXP_UNIQUE_ID", "EXP_STRATEGY", "EXP_DATASET"]
    ]
    plot_df = Base_Visualizer.load_detailed_metric_files(
        done_workload_df, metric, OUTPUT_PATH
    )

    if plot_df.empty:
        return "Dataframe was empty"

    del plot_df["EXP_UNIQUE_ID"]

    plot_df = plot_df.melt(
        id_vars=["EXP_STRATEGY", "EXP_DATASET"], var_name="AL Cycle", value_name=metric
    )

    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={"metric": metric},
        plot_function=_plot_function,
        df_col_key="EXP_DATASET",
        legend_names=plot_df["EXP_STRATEGY"].unique(),
    )
    return plot_urls


class Learning_Curves(Base_Visualizer):
    @staticmethod
    def get_additional_request_params(
        OUTPUT_PATH: Path, with_basic=True
    ) -> Dict[str, List[Any]]:
        smm = Standard_ML_Metrics()
        return {"VIZ_LC_METRIC": smm.metrics}

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_LC_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_LC_METRIC value"}

        metric = self._exp_grid_request_params["VIZ_LC_METRIC"][0]
        # read in all metrics
        done_workload_df = self._load_done_workload()

        plot_url = _cache_create_plots(
            metric=metric,
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
        )

        return {"plot_data": plot_url}
