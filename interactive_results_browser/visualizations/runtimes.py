from __future__ import annotations
import base64
from distutils.command.config import config
import io

import pandas as pd
from datasets import DATASET

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

from resources.data_types import AL_STRATEGY, LEARNER_MODEL

if TYPE_CHECKING:
    from misc.config import Config
import seaborn as sns
import matplotlib.pyplot as plt

from interactive_results_browser.cache import memory


@memory.cache()
def _cache_runtimes(_exp_grid_request_params, done_workload) -> str:
    if len(_exp_grid_request_params["VIZ_RT_METRIC"]) != 1:
        return {"ERROR": "Please select only one VIZ_RT_METRIC value"}

    metric = _exp_grid_request_params["VIZ_RT_METRIC"][0]

    rel = sns.FacetGrid(
        done_workload,
        col="EXP_DATASET",
        col_wrap=min(6, len(_exp_grid_request_params["EXP_DATASET"])),
        # subplot_kws=dict(margin_titles=True),
    )
    rel.map_dataframe(
        sns.barplot, x=metric, y="EXP_STRATEGY", orient="h"
    )  # , hue="EXP_STRATEGY")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode("utf8")
    return plot_url


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
        plot_url = _cache_runtimes(
            _exp_grid_request_params=self._exp_grid_request_params,
            done_workload=self._load_done_workload(),
        )
        return {"plot_data": plot_url}
