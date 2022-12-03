from __future__ import annotations
import base64
from distutils.command.config import config
import io

import pandas as pd
from datasets import DATASET

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

from resources.data_types import AL_STRATEGY, LEARNER_MODEL

if TYPE_CHECKING:
    from misc.config import Config
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
        style="Strategy",
        # markers="Strategy",
        markers=my_markers,
        palette=my_palette,
        hue="Strategy",
    )
    rel.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    rel.set(ylim=(0, 1))
    return rel


@memory.cache()
def _cache_create_plots(
    done_workload_df,
    metric,
    OUTPUT_PATH,
    METRIC_RESULTS_PATH_APPENDIX,
    max_col_wrap,
) -> List[str]:
    result_data = []
    # read in each csv file to get learning curve data for plot
    for _, row in done_workload_df.iterrows():
        detailed_metrics_df = pd.read_csv(
            f"{OUTPUT_PATH}/{row['EXP_DATASET']}/{row['EXP_UNIQUE_ID']}{METRIC_RESULTS_PATH_APPENDIX}",
            usecols=[metric],
        )

        for ix, row2 in detailed_metrics_df.iterrows():
            result_data.append(
                (row["EXP_STRATEGY"], row["EXP_DATASET"], str(ix), row2[metric])
            )

    plot_df = pd.DataFrame(
        data=result_data, columns=["Strategy", "Dataset", "AL Cycle", metric]
    )

    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={"metric": metric},
        plot_function=_plot_function,
        df_col_key="Dataset",
        legend_names=plot_df["Strategy"].unique(),
    )
    return plot_urls


class Learning_Curves(Base_Visualizer):
    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
        return {
            "VIZ_LC_METRIC": [
                "accuracy",
                # "precision", TODO <- die metriken gibt es je einmal pro klasse -> wie darstellen?
                # "f1-score",
                # "support",
                # "recall",
                "macro avg_precision",
                "macro avg_recall",
                "macro avg_f1-score",
                "macro avg_support",
                "weighted avg_precision",
                "weighted avg_recall",
                "weighted avg_f1-score",
                "weighted avg_support",
            ]
        }

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_LC_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_LC_METRIC value"}

        metric = self._exp_grid_request_params["VIZ_LC_METRIC"][0]
        # read in all metrics
        done_workload_df = self._load_done_workload()

        plot_url = _cache_create_plots(
            max_col_wrap=len(self._exp_grid_request_params["EXP_DATASET"]),
            metric=metric,
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
            METRIC_RESULTS_PATH_APPENDIX=self._config.METRIC_RESULTS_PATH_APPENDIX,
        )

        return {"plot_data": plot_url}
