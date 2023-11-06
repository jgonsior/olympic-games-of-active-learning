from __future__ import annotations
import glob
import math
from pathlib import Path

import pandas as pd


from analyse_results.visualizations.base_visualizer import (
    MERGE_AL_CYCLE_METRIC_STRATEGY,
    Base_Visualizer,
)
from typing import Any, List

from typing import Any, Dict

from analyse_results.cache import memory

from misc.config import Config

# if TYPE_CHECKING:
import seaborn as sns
import matplotlib.pyplot as plt


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
    done_workload_df: pd.DataFrame,
    OUTPUT_PATH: Path,
    config: Config,
    metric_values: List[str],
) -> List[str]:
    done_workload_df = done_workload_df.loc[
        :, ["EXP_UNIQUE_ID", "EXP_STRATEGY", "EXP_DATASET"]
    ]

    plot_df = pd.DataFrame()

    for metric in metric_values:
        single_metric_plot_df = Base_Visualizer.load_detailed_metric_files(
            done_workload_df,
            metric,
            OUTPUT_PATH,
            merge_al_cycle_metrics=MERGE_AL_CYCLE_METRIC_STRATEGY.ORIGINAL_MEAN,
        )

        column_names_which_are_al_cycles = [
            c
            for c in single_metric_plot_df.columns.to_list()
            if not c.startswith("EXP_")
        ]

        if len(single_metric_plot_df) == 0:
            print(f"No data for  metric {metric} found")
            continue

        if (
            max(
                [
                    a[0]
                    for a in single_metric_plot_df[
                        column_names_which_are_al_cycles
                    ].values
                ]
            )
            <= 1.0
        ):
            single_metric_plot_df[
                column_names_which_are_al_cycles
            ] = single_metric_plot_df[column_names_which_are_al_cycles].multiply(100)

        if len(plot_df) == 0:
            plot_df = single_metric_plot_df
        else:
            plot_df = plot_df.merge(
                single_metric_plot_df,
                on=["EXP_UNIQUE_ID", "EXP_DATASET", "EXP_STRATEGY"],
                how="inner",
            )
            plot_df.drop_duplicates(inplace=True)

    if len(plot_df) == 0:
        return "No data found to plot"

    del plot_df["EXP_UNIQUE_ID"]

    plot_df = plot_df.melt(
        id_vars=["EXP_STRATEGY", "EXP_DATASET"],
        var_name="metric",
        value_vars=column_names_which_are_al_cycles,
    )
    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={},
        plot_function=_plot_function_auc,
        df_col_key="metric",
        legend_names=column_names_which_are_al_cycles,
        create_legend=False,
    )

    return plot_urls


class Auc_Table(Base_Visualizer):
    @staticmethod
    def get_additional_request_params(
        OUTPUT_PATH: Path, with_basic=True
    ) -> Dict[str, List[Any]]:
        all_existing_metric_names = set(
            [Path(a).name for a in glob.glob(str(OUTPUT_PATH / "*/*/*.csv.xz"))]
        )
        all_existing_metric_names = [
            a.split(".")[0]
            for a in all_existing_metric_names
            if not a.startswith("auc_")
            and not a.startswith("learning_stability_")
            and not a.startswith("pickled_learner_model")
        ]

        return {"VIZ_AUC_TABLE_METRIC": sorted(all_existing_metric_names)}

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_AUC_TABLE_METRIC"]) == 0:
            return {"ERROR": "Please select at least one VIZ_AUC_TABLE_METRIC value"}

        # read in all metrics
        done_workload_df = self._load_done_workload()

        plot_url = _cache_create_auc_table(
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
            config=self._config,
            metric_values=self._exp_grid_request_params["VIZ_AUC_TABLE_METRIC"],
        )

        return {"plot_data": plot_url}
