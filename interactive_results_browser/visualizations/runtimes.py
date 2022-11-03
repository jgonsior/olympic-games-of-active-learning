from __future__ import annotations
from distutils.command.config import config

import pandas as pd
from datasets import DATASET

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

from resources.data_types import AL_STRATEGY, LEARNER_MODEL

if TYPE_CHECKING:
    from misc.config import Config
import plotly
import plotly.express as px
import plotly.graph_objs as go
import json


class Runtimes(Base_Visualizer):
    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
        return {
            "VIZ_RT_METRIC": [
                "learner_training_time",
                "query_selection_time",
            ]
        }

    def _load_done_workload(self, limit_to_get_params=True) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self._config.DONE_WORKLOAD_PATH)

        if limit_to_get_params:
            for k, v in self._exp_grid_request_params.items():
                if k in self._NON_WORKLOAD_KEYS:
                    continue
                df = df.loc[df[k].isin(v)]

        # convert int_enums to real enums
        df["EXP_DATASET"] = df["EXP_DATASET"].apply(lambda x: DATASET(x).name)
        df["EXP_LEARNER_MODEL"] = df["EXP_LEARNER_MODEL"].apply(
            lambda x: LEARNER_MODEL(x).name
        )
        df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(lambda x: AL_STRATEGY(x).name)
        return df

    def get_template_data(self) -> Dict[str, Any]:
        # read in all metrics
        df = self._load_done_workload()

        if (
            "VIZ_RT_METRIC" not in self._exp_grid_request_params
            or len(self._exp_grid_request_params["VIZ_RT_METRIC"]) != 1
        ):
            return {"ERROR": "Please select only one VIZ_RT_METRIC value"}

        number_facets = int(len(df["EXP_DATASET"].unique()) / 3)

        fig = px.bar(
            df,
            x="EXP_STRATEGY",
            y=self._exp_grid_request_params["VIZ_LC_METRIC"][0],
            color="EXP_STRATEGY",
            pattern_shape="EXP_STRATEGY",
            text_auto=True,
            facet_col="EXP_DATASET",
            facet_col_wrap=6,
            width=3000,
            height=number_facets * 100,
            facet_row_spacing=0.02,
            facet_col_spacing=0.01,
        )

        fig.update_yaxes(title=None)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return {"graphJSON": graphJSON}
