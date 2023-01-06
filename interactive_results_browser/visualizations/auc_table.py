from __future__ import annotations
import math
from pathlib import Path

import pandas as pd


from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, List, TYPE_CHECKING

from typing import Any, Dict

from interactive_results_browser.cache import memory
from metrics.computed.STANDARD_AUC import STANDARD_AUC

if TYPE_CHECKING:
    from misc.config import Config
import seaborn as sns
import matplotlib.pyplot as plt


@memory.cache()
def _plot_function_auc(plot_df, my_palette, my_markers):
    fig, ax = plt.subplots(
        figsize=(
            len(plot_df["EXP_STRATEGY"].unique()),
            math.ceil(len(plot_df["EXP_DATASET"].unique()) / 3),
        )
    )
    del plot_df["metric"]
    plot_df = plot_df.groupby(by=["EXP_STRATEGY", "EXP_DATASET"], as_index=False).mean()
    plot_df = plot_df.pivot(index="EXP_DATASET", columns="EXP_STRATEGY", values="value")

    # sort
    plot_df = plot_df.sort_index(axis=0)
    plot_df = plot_df.reindex(sorted(plot_df.columns), axis=1)

    # new row with mean (%) and mean(r)
    plot_df.loc["Mean (%)"] = plot_df.mean(axis=0)
    ordered_columns = (
        plot_df.loc["Mean (%)"].sort_values(ascending=False).index.tolist()
    )
    plot_df = plot_df[ordered_columns]

    ax = sns.heatmap(
        data=plot_df,
        # square=True,
        annot=True,
        cmap=sns.color_palette("coolwarm_r", as_cmap=True),
        fmt=".2f",
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        # vmin=0,
        # vmax=100,
        linewidth=0.5,
    )
    ax.set(ylabel=None)
    ax.set(xlabel=None)

    ax.xaxis.tick_top()

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        horizontalalignment="left",
    )
    return ax


@memory.cache()
def _cache_create_auc_table(
    done_workload_df: pd.DataFrame, OUTPUT_PATH: Path, config: Config
) -> List[str]:
    done_workload_df = done_workload_df.loc[
        :, ["EXP_UNIQUE_ID", "EXP_STRATEGY", "EXP_DATASET"]
    ]
    sacmc = STANDARD_AUC(config)
    metric_values = [
        sacmc.computed_metric_appendix() + "_" + sss for sss in sacmc.metrics
    ]

    metric_values += [
        "avg_dist_batch",
        "avg_dist_labeled",
        "avg_dist_unlabeled",
        "mismatch_train_test",
    ]

    plot_df = pd.DataFrame()

    for metric in metric_values:
        single_metric_plot_df = Base_Visualizer.load_detailed_metric_files(
            done_workload_df, metric, OUTPUT_PATH
        )

        single_metric_plot_df[metric] = single_metric_plot_df[
            "computed_metric"
        ].multiply(100)
        del single_metric_plot_df["computed_metric"]

        if len(plot_df) == 0:
            plot_df = single_metric_plot_df
        else:
            plot_df = plot_df.merge(
                single_metric_plot_df,
                on=["EXP_UNIQUE_ID", "EXP_DATASET", "EXP_STRATEGY"],
                how="inner",
            )

    del plot_df["EXP_UNIQUE_ID"]
    plot_df = plot_df.melt(
        id_vars=["EXP_STRATEGY", "EXP_DATASET"],
        var_name="metric",
        value_vars=metric_values,
    )
    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={},
        plot_function=_plot_function_auc,
        df_col_key="metric",
        legend_names=metric_values,
        create_legend=False,
    )

    return plot_urls


class Auc_Table(Base_Visualizer):
    def get_template_data(self) -> Dict[str, Any]:
        # read in all metrics
        done_workload_df = self._load_done_workload()

        plot_url = _cache_create_auc_table(
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
            config=self._config,
        )

        return {"plot_data": plot_url}
