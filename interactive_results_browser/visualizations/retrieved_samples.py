from __future__ import annotations
import ast
from collections import Counter
import itertools

import numpy as np
import pandas as pd

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config
import seaborn as sns
import matplotlib.pyplot as plt
from interactive_results_browser.cache import memory


def _plot_function(plot_df, my_palette, my_markers):
    # fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = plot_df.pivot(index="STRAT_A", columns="STRAT_B", values="JACCARD")

    # sort
    plot_df = plot_df.sort_index(axis=0)
    plot_df = plot_df.reindex(sorted(plot_df.columns), axis=1)

    ax = sns.heatmap(
        data=plot_df,
        annot=True,
        cmap=sns.color_palette("Spectral", as_cmap=True),
        fmt=".2f",
        # ax=ax,
        xticklabels=True,
        yticklabels=True,
        vmin=0,
        vmax=1,
    )
    ax.set(ylabel=None)
    ax.set(xlabel=None)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        horizontalalignment="right",
    )
    return ax


# @memory.cache()
def _cache_retrieved_samples(done_workload) -> List[str]:
    plot_df = done_workload.filter(
        [
            "selected_indices",
            "EXP_STRATEGY",
            "EXP_DATASET",
        ]
    )
    plot_df["selected_indices"] = plot_df["selected_indices"].apply(ast.literal_eval)

    # remove starting point
    plot_df["selected_indices"] = plot_df["selected_indices"].apply(
        lambda x: [
            single_value for single_list in x[1:] for single_value in single_list
        ]
    )

    new_data = []

    # create large count_dict per dataset and strategy, and calculate, how often each sample was queried
    for dataset in plot_df["EXP_DATASET"].unique():
        for (strat_a, strat_b) in itertools.combinations_with_replacement(
            plot_df["EXP_STRATEGY"].unique(), 2
        ):
            sampled_indices_strat_a = [
                single_value
                for single_list in plot_df.loc[
                    (plot_df["EXP_DATASET"] == dataset)
                    & (plot_df["EXP_STRATEGY"] == strat_a)
                ]["selected_indices"].to_list()
                for single_value in single_list
            ]

            sampled_indices_strat_b = [
                single_value
                for single_list in plot_df.loc[
                    (plot_df["EXP_DATASET"] == dataset)
                    & (plot_df["EXP_STRATEGY"] == strat_b)
                ]["selected_indices"].to_list()
                for single_value in single_list
            ]

            if len(sampled_indices_strat_a) == 0 or len(sampled_indices_strat_b) == 0:
                jaccard = -1
            else:
                counter_a = Counter(sampled_indices_strat_a)
                counter_b = Counter(sampled_indices_strat_b)
                # note: we include duplicates!
                intersection_count = len(list((counter_a & counter_b).elements()))
                union_count = len(list((counter_a | counter_b).elements()))

                jaccard = intersection_count / union_count
            new_data.append([dataset, strat_a, strat_b, jaccard])

            # also add strat_b,strat_a as result
            if strat_a != strat_b:
                new_data.append([dataset, strat_b, strat_a, jaccard])
    new_data = pd.DataFrame(
        new_data, columns=["EXP_DATASET", "STRAT_A", "STRAT_B", "JACCARD"]
    )

    plot_urls = Base_Visualizer._render_images(
        plot_df=new_data,
        args={},
        plot_function=_plot_function,
        legend_names=done_workload["EXP_STRATEGY"].unique(),
        df_col_key="EXP_DATASET",
        create_legend=False,
    )
    return plot_urls


class Retrieved_Samples(Base_Visualizer):
    def get_template_data(self) -> Dict[str, Any]:
        plot_url = _cache_retrieved_samples(
            done_workload=self._load_done_workload(),
        )
        return {"plot_data": plot_url}
