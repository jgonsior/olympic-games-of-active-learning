from __future__ import annotations
from distutils.command.config import config

import pandas as pd

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config
import plotly
import plotly.express as px
import json


class Learning_Curves(Base_Visualizer):
    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
        return {
            "VIZ_LC_METRIC": [
                "learner_training_time",
                "query_selection_time",
                "acc_auc",
                "macro_f1_auc",
                "macro_prec_auc",
                "macro_recall_auc",
                "weighted_f1_auc",
                "weighted_prec_auc",
                "weighted_recall_auc",
            ]
        }

    def _load_done_workload(self, limit_to_get_params=True) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self._config.DONE_WORKLOAD_PATH)

        if limit_to_get_params:
            for k, v in self._exp_grid_request_params.items():
                if k in self._NON_WORKLOAD_KEYS:
                    continue
                df = df.loc[df[k].isin(v)]
        return df

    def get_template_data(self) -> Dict[str, Any]:
        # read in all metrics
        done_workload = self._load_done_workload()

        if len(self._exp_grid_request_params["VIZ_LC_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_LC_METRIC value"}

        print(done_workload[self._exp_grid_request_params["VIZ_LC_METRIC"][0]])

        fig = px.bar(
            done_workload,
            x="EXP_STRATEGY",
            y=self._exp_grid_request_params["VIZ_LC_METRIC"][0],
        )

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return {"graphJSON": graphJSON}
