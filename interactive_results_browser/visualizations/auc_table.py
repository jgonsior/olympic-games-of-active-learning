from __future__ import annotations
import copy
import math
from pathlib import Path

import pandas as pd


from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, List

from typing import Any, Dict

from interactive_results_browser.cache import memory
from metrics.computed.STANDARD_AUC import STANDARD_AUC

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
            done_workload_df, metric, OUTPUT_PATH
        )

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
    @staticmethod
    def get_additional_request_params(with_basic=True) -> Dict[str, List[Any]]:
        sacmc = STANDARD_AUC(Config())
        basic_metrics = sacmc.metrics
        auc_metrics = []
        auc_ranges = [
            "ramp_up_quality_",
            "middle_quality_",
            "end_quality_",
            "learning_stability_",
            "auc_",
        ]
        for ar in auc_ranges:
            auc_metrics = [*auc_metrics, *[ar + sss for sss in basic_metrics]]

        metric_values = copy.deepcopy(auc_metrics)

        for basic_metric in basic_metrics:
            metric_values.append("biggest_drop_per_" + basic_metric)
            metric_values.append("nr_decreasing_al_cycles_per_" + basic_metric)
            if with_basic:
                metric_values.append(basic_metric)

        metric_values += [
            "avg_dist_batch",
            "avg_dist_labeled",
            "avg_dist_unlabeled",
            "mismatch_train_test",
            "class_distributions_chebyshev",
            "class_distributions_manhattan",
            "hardest_samples",
            "optimal_samples_order_wrongness",
            "optimal_samples_order_variability",
            "optimal_samples_order_easy_hard_ambiguous",
            "optimal_samples_order_acc_diff_addition",
            "optimal_samples_order_acc_diff_absolute_addition",
            "optimal_samples_included_in_optimal_strategy",
        ]

        for standard_metric_without_auc in basic_metrics:
            if standard_metric_without_auc in metric_values:
                metric_values.remove(standard_metric_without_auc)
        return {"VIZ_AUC_TABLE_METRIC": metric_values}

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
