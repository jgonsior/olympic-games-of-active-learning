from __future__ import annotations
from pathlib import Path

import pandas as pd

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, List, TYPE_CHECKING

from typing import Any, Dict


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

    plot_df = pd.DataFrame()

    for EXP_STRATEGY in done_workload_df["EXP_STRATEGY"].unique():
        for EXP_DATASET in done_workload_df["EXP_DATASET"].unique():
            detailed_metrics_path = Path(
                f"{OUTPUT_PATH}/{EXP_STRATEGY}/{EXP_DATASET}/{metric}.csv.gz"
            )
            if detailed_metrics_path.exists():
                # read in each csv file to get learning curve data for plot
                detailed_metrics_df = pd.read_csv(detailed_metrics_path)

                detailed_metrics_df = detailed_metrics_df.merge(
                    done_workload_df, on="EXP_UNIQUE_ID", how="inner"
                )
                plot_df = pd.concat([plot_df, detailed_metrics_df], ignore_index=True)

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
            metric=metric,
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
        )

        return {"plot_data": plot_url}
