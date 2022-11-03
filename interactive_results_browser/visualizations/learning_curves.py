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

        if len(self._exp_grid_request_params["VIZ_LC_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_LC_METRIC value"}

        g = sns.relplot(
            df,
            x="EXP_STRATEGY",
            y=self._exp_grid_request_params["VIZ_LC_METRIC"][0],
            col="EXP_DATASET",
            col_wrap=6,
        )

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode("utf8")

        return {"plot_data": plot_url}
