from __future__ import annotations


from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, List, TYPE_CHECKING

from typing import Any, Dict

from interactive_results_browser.cache import memory

if TYPE_CHECKING:
    pass
import seaborn as sns
import matplotlib.pyplot as plt


def _plot_function(plot_df, my_palette, my_markers):
    fig, ax = plt.subplots(
        figsize=(
            len(plot_df["EXP_STRATEGY"].unique()),
            int(len(plot_df["EXP_DATASET"].unique()) / 3),
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
    done_workload_df,
) -> List[str]:
    metric_values = [
        "acc_auc",
        "macro_f1_auc",
        "macro_prec_auc",
        "macro_recall_auc",
        "weighted_f1_auc",
        "weighted_prec_auc",
        "weighted_recall_auc",
    ]

    plot_df = done_workload_df.filter([*metric_values, "EXP_STRATEGY", "EXP_DATASET"])
    plot_df = plot_df.melt(
        id_vars=["EXP_STRATEGY", "EXP_DATASET"], var_name="metric", value_name="value"
    )

    plot_df["value"] = plot_df["value"].multiply(100)

    plot_urls = Base_Visualizer._render_images(
        plot_df=plot_df,
        args={},
        plot_function=_plot_function,
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
        )

        return {"plot_data": plot_url}
