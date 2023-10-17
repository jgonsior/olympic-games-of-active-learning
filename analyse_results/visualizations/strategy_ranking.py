from __future__ import annotations
import math
from pathlib import Path

import pandas as pd
from analyse_results.visualizations.auc_table import Auc_Table


from analyse_results.visualizations.base_visualizer import (
    MERGE_AL_CYCLE_METRIC_STRATEGY,
    Base_Visualizer,
)
from typing import Any, Dict, List

from misc.config import Config

import seaborn as sns
import matplotlib.pyplot as plt


def _plot_correlation_analysis(plot_df, my_palette, my_markers):
    fig, ax = plt.subplots(
        figsize=(
            len(plot_df.columns),
            math.ceil(len(plot_df.columns.unique()) / 3),
        )
    )

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


def _plot_strategy_ranking_heatmap(plot_df, my_palette, my_markers):
    del plot_df["EXP_DATASET"]

    plot_df = plot_df.groupby(
        by=["EXP_STRATEGY"],
    ).mean()

    # normalize columns
    plot_df = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())

    fig, ax = plt.subplots(
        figsize=(
            len(plot_df.columns),
            math.ceil(len(plot_df) / 2),
        )
    )

    ax = sns.heatmap(
        data=plot_df,
        # square=True,
        annot=True,
        cmap=sns.color_palette("Spectral", as_cmap=True),
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


def _correlation_analysis(done_workload_df, OUTPUT_PATH):
    plot_df = pd.DataFrame()
    metric_values = Auc_Table.get_additional_request_params(
        OUTPUT_PATH, with_basic=True
    )["VIZ_AUC_TABLE_METRIC"]

    for metric in metric_values:
        single_metric_plot_df = Base_Visualizer.load_detailed_metric_files(
            done_workload_df,
            metric,
            OUTPUT_PATH,
            merge_al_cycle_metrics=MERGE_AL_CYCLE_METRIC_STRATEGY.MEAN_LIST,
        )
        print(done_workload_df)
        print(metric_values)
        print(metric)
        print(single_metric_plot_df)

        if len(single_metric_plot_df) == 0:
            print(f"No data for  metric {metric} found")
            continue

        if single_metric_plot_df["computed_metric"].max() <= 1.0:
            single_metric_plot_df[metric] = single_metric_plot_df[
                "computed_metric"
            ].multiply(100)
        else:
            single_metric_plot_df[metric] = single_metric_plot_df["computed_metric"]
        del single_metric_plot_df["computed_metric"]

        if len(plot_df) == 0:
            plot_df = single_metric_plot_df
        else:
            plot_df = plot_df.merge(
                single_metric_plot_df,
                on=["EXP_UNIQUE_ID", "EXP_DATASET", "EXP_STRATEGY"],
                how="inner",
            )
            plot_df.drop_duplicates(inplace=True)
    del plot_df["EXP_UNIQUE_ID"]
    del plot_df["EXP_DATASET"]
    plot_df = plot_df.groupby(by=["EXP_STRATEGY"]).mean()
    plot_df = plot_df.reindex(sorted(plot_df.columns), axis=1)
    plot_df = plot_df.corr(method="spearman")

    return plot_df


def _strategy_ranking_heatmap(done_workload_df, OUTPUT_PATH):
    plot_df = pd.DataFrame()
    metric_values = Auc_Table.get_additional_request_params(
        OUTPUT_PATH, with_basic=True
    )["VIZ_AUC_TABLE_METRIC"]
    for metric in metric_values:
        single_metric_plot_df = Base_Visualizer.load_detailed_metric_files(
            done_workload_df, metric, OUTPUT_PATH
        )
        if len(single_metric_plot_df) == 0:
            print(f"No data for  metric {metric} found.")
            continue
        print(metric)
        print(single_metric_plot_df)
        if single_metric_plot_df["computed_metric"].max() <= 1.0:
            single_metric_plot_df[metric] = single_metric_plot_df[
                "computed_metric"
            ].multiply(100)
        else:
            single_metric_plot_df[metric] = single_metric_plot_df["computed_metric"]
        del single_metric_plot_df["computed_metric"]

        if len(plot_df) == 0:
            plot_df = single_metric_plot_df
        else:
            plot_df = plot_df.merge(
                single_metric_plot_df,
                on=["EXP_UNIQUE_ID", "EXP_DATASET", "EXP_STRATEGY"],
                how="inner",
            )
            plot_df.drop_duplicates(inplace=True)

    # per dataset
    # averaged over all datasets
    # ranking number
    del plot_df["EXP_UNIQUE_ID"]

    datasets = plot_df["EXP_DATASET"].unique().tolist()
    datasets.append("mean")
    datasets.append("rank")

    return plot_df


# @memory.cache()
def _cache_strategy_ranking(
    done_workload_df: pd.DataFrame,
    OUTPUT_PATH: Path,
    config: Config,
    metric_values: List[str],
) -> List[str]:
    done_workload_df = done_workload_df.loc[
        :, ["EXP_UNIQUE_ID", "EXP_STRATEGY", "EXP_DATASET"]
    ]

    # calculate over all selected metrics the rankings
    if metric_values == ["correlation_analysis"]:
        plot_df = _correlation_analysis(done_workload_df, OUTPUT_PATH)

        plot_urls = Base_Visualizer._render_images(
            plot_df=plot_df,
            args={},
            plot_function=_plot_correlation_analysis,
            df_col_key=None,
            legend_names=metric_values,
            create_legend=False,
        )
    elif metric_values == ["strategy_ranking_heatmap"]:
        plot_df = _strategy_ranking_heatmap(done_workload_df, OUTPUT_PATH)

        plot_urls = Base_Visualizer._render_images(
            plot_df=plot_df,
            args={},
            plot_function=_plot_strategy_ranking_heatmap,
            df_col_key="EXP_DATASET",
            legend_names=metric_values,
            create_legend=False,
            combined_df_col_key_plot=True,
        )

    return plot_urls


class Strategy_Ranking(Base_Visualizer):
    @staticmethod
    def get_additional_request_params(
        OUTPUT_PATH: Path, with_basic=True
    ) -> Dict[str, List[Any]]:
        possible_metrics = [
            "correlation_analysis",
            "strategy_ranking_heatmap",
        ]

        return {"VIZ_RANKING_METHOD": possible_metrics}

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_RANKING_METHOD"]) == 0:
            return {"ERROR": "Please select at least one VIZ_RANKING_METHOD value"}

        # read in all metrics
        done_workload_df = self._load_done_workload()

        plot_url = _cache_strategy_ranking(
            done_workload_df=done_workload_df,
            OUTPUT_PATH=self._config.OUTPUT_PATH,
            config=self._config,
            metric_values=self._exp_grid_request_params["VIZ_RANKING_METHOD"],
        )

        return {"plot_data": plot_url}
