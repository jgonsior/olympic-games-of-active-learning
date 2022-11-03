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
from matplotlib.ticker import FuncFormatter


class Learning_Curves(Base_Visualizer):
    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
        return {
            "VIZ_LC_METRIC": [
                "accuracy",
                # "precision", TODO <- die metriken gibt es je einmal pro klasse -> wie darstellen?
                # "f1-score",
                # "support",
                # "recall",
                "macro avg_precision",
                "macro avg_recall",
                "macro avg_f1-score",
                "macro avg_support",
                "weighted avg_precision",
                "weighted avg_recall",
                "weighted avg_f1-score",
                "weighted avg_support",
            ]
        }

    def _load_done_workload(
        self, limit_to_get_params=True, enum_to_str=True
    ) -> pd.DataFrame:
        df: pd.DataFrame = pd.read_csv(self._config.DONE_WORKLOAD_PATH)

        if limit_to_get_params:
            for k, v in self._exp_grid_request_params.items():
                if k in self._NON_WORKLOAD_KEYS:
                    continue
                df = df.loc[df[k].isin(v)]

        if enum_to_str:
            # convert int_enums to real enums
            df["EXP_DATASET"] = df["EXP_DATASET"].apply(lambda x: DATASET(x).name)
            df["EXP_LEARNER_MODEL"] = df["EXP_LEARNER_MODEL"].apply(
                lambda x: LEARNER_MODEL(x).name
            )
            df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(lambda x: AL_STRATEGY(x).name)
        return df

    def get_template_data(self) -> Dict[str, Any]:
        if len(self._exp_grid_request_params["VIZ_LC_METRIC"]) != 1:
            return {"ERROR": "Please select only one VIZ_LC_METRIC value"}

        metric = self._exp_grid_request_params["VIZ_LC_METRIC"][0]
        # read in all metrics
        done_workload_df = self._load_done_workload()

        result_data = []
        # read in each csv file to get learning curve data for plot
        for _, row in done_workload_df.iterrows():
            detailed_metrics_df = pd.read_csv(
                f"{self._config.OUTPUT_PATH}/{row['EXP_DATASET']}/{row['EXP_UNIQUE_ID']}{self._config.METRIC_RESULTS_PATH_APPENDIX}",
                usecols=[metric],
            )

            for ix, row2 in detailed_metrics_df.iterrows():
                result_data.append(
                    (row["EXP_STRATEGY"], row["EXP_DATASET"], str(ix), row2[metric])
                )

        results = pd.DataFrame(
            data=result_data, columns=["Strategy", "Dataset", "AL Cycle", metric]
        )
        rel = sns.relplot(
            results,
            x="AL Cycle",
            y=metric,
            hue="Strategy",
            kind="line",
            style="Strategy",
            col="Dataset",
            markers=True,
            col_wrap=min(6, len(self._exp_grid_request_params["EXP_DATASET"])),
        )
        for ax in rel.fig.axes:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode("utf8")

        return {"plot_data": plot_url}
