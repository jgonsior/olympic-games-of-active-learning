from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import io
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List

from typing import Any, Dict

from flask import render_template
from joblib import Parallel, delayed, parallel_backend
import matplotlib.pyplot as plt
import pandas as pd
from datasets import DATASET
from interactive_results_browser.cache import memory
from resources.data_types import AL_STRATEGY, LEARNER_MODEL
import seaborn as sns
from seaborn._core.properties import Marker

if TYPE_CHECKING:
    from misc.config import Config


@memory.cache
def _cache_load_done_workload(
    done_workload_path,
    limit_to_get_params,
    enum_to_str,
    non_workload_keys,
    exp_grid_request_params,
) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(done_workload_path, engine="pyarrow")

    if limit_to_get_params:
        for k, v in exp_grid_request_params.items():
            if k in non_workload_keys:
                continue
            if k in df.columns:
                df = df.loc[df[k].isin(v)]

    if enum_to_str:
        # convert int_enums to real enums
        df["EXP_DATASET"] = df["EXP_DATASET"].apply(lambda x: DATASET(x).name)
        df["EXP_LEARNER_MODEL"] = df["EXP_LEARNER_MODEL"].apply(
            lambda x: LEARNER_MODEL(x).name
        )
        df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(lambda x: AL_STRATEGY(x).name)
    return df


class Base_Visualizer(ABC):
    _NON_WORKLOAD_KEYS = ["VISUALIZATIONS"]

    def __init__(
        self,
        config: Config,
        exp_grid_request_params: Dict[str, Any],
        experiment_name: str,
    ) -> None:
        self._config = config
        self._exp_grid_request_params = exp_grid_request_params
        self._experiment_name = experiment_name

        additional_request_params = self.__class__.get_additional_request_params()

        for k in additional_request_params.keys():
            if not k in self._NON_WORKLOAD_KEYS:
                self._NON_WORKLOAD_KEYS.append(k)

            # in case nothing has been selected in the gui just take the first parameter as default
            if k not in self._exp_grid_request_params.keys():
                self._exp_grid_request_params[k] = [additional_request_params[k][0]]

    def get_template_name(self) -> str:
        return "partials/" + self.__class__.__name__.lower() + ".html.j2"

    @abstractmethod
    def get_template_data(self) -> Dict[str, Any]:
        return {}

    def render(self) -> str:
        result = render_template(self.get_template_name(), **self.get_template_data())
        plt.clf()
        return result

    @staticmethod
    def get_additional_request_params() -> Dict[str, List[Any]]:
        return {}

    @staticmethod
    def __render_single_image_multithreaded(
        plot_function: Callable[[Any], Any],
        plot_df: pd.DataFrame,
        my_color_dict: Dict[str, str],
        my_markers: Dict[str, Any],
        args: Dict[str, Any],
        df_col_value: str,
        df_col_key: str,
    ):
        if df_col_value != "ALL_DATASETS":
            plot_df = plot_df.loc[plot_df[df_col_key] == df_col_value]
        args["my_palette"] = my_color_dict
        args["my_markers"] = my_markers
        ax = plot_function(plot_df, **args)

        if ax != None:
            ax.set(title=df_col_value)

        plt.legend([], [], frameon=False)
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode("utf8")
        plt.clf()
        return plot_url

    @staticmethod
    def _render_images(
        plot_df: pd.DataFrame,
        args: Dict[str, Any],
        plot_function: Callable[[Any], Any],
        df_col_key: str,
        legend_names: List[str],
        create_legend: bool = True,
        combined_df_col_key_plot: bool = False,
    ) -> List[str]:
        legend_names = sorted(legend_names)
        # calculate colormap
        palette_colors = sns.color_palette("husl", len(legend_names))
        my_color_dict = {k: v for k, v in zip(legend_names, palette_colors)}

        # calculate markers

        marker_values = Marker()
        marker_values = marker_values._default_values(len(legend_names))
        my_markers = {k: v for k, v in zip(legend_names, marker_values)}

        df_col_values = plot_df[df_col_key].unique().tolist()
        if combined_df_col_key_plot:
            df_col_values.append("ALL_DATASETS")

        with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
            figs = Parallel()(
                delayed(Base_Visualizer.__render_single_image_multithreaded)(
                    plot_function=plot_function,
                    plot_df=plot_df,
                    args=args,
                    my_color_dict=my_color_dict,
                    my_markers=my_markers,
                    df_col_value=df_col_value,
                    df_col_key=df_col_key,
                )
                for df_col_value in df_col_values
            )

        if create_legend:
            legend_df = pd.DataFrame(
                [(k, v) for k, v in my_color_dict.items()], columns=["label", "color"]
            )
            ax = sns.lineplot(
                legend_df,
                legend=True,
                x="label",
                y=[1 for _ in range(0, len(legend_df))],
                hue="label",
                palette=my_color_dict,
                markers=my_markers,
                style="label",
            )

            fig2 = plt.figure()
            ax2 = fig2.add_subplot()
            ax2.axis("off")
            legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False)
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted(),
            )

            img = io.BytesIO()
            plt.savefig(img, format="png", bbox_inches=bbox)
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode("utf8")
            figs = [plot_url, *figs]
        return figs

    def _load_done_workload(
        self, limit_to_get_params=True, enum_to_str=True
    ) -> pd.DataFrame:
        return _cache_load_done_workload(
            done_workload_path=self._config.OVERALL_DONE_WORKLOAD_PATH,
            limit_to_get_params=limit_to_get_params,
            enum_to_str=enum_to_str,
            non_workload_keys=self._NON_WORKLOAD_KEYS,
            exp_grid_request_params=self._exp_grid_request_params,
        )

    @staticmethod
    def load_detailed_metric_files(
        done_workload_df: pd.DataFrame,
        metric: str,
        OUTPUT_PATH: Path,
    ) -> pd.DataFrame:
        result_df = pd.DataFrame()

        for EXP_STRATEGY in done_workload_df["EXP_STRATEGY"].unique():
            for EXP_DATASET in done_workload_df["EXP_DATASET"].unique():
                detailed_metrics_path = Path(
                    f"{OUTPUT_PATH}/{EXP_STRATEGY}/{EXP_DATASET}/{metric}.csv.xz"
                )

                if detailed_metrics_path.exists():
                    # read in each csv file to get learning curve data for plot
                    detailed_metrics_df = pd.read_csv(detailed_metrics_path, engine="pyarrow")

                    detailed_metrics_df = detailed_metrics_df.merge(
                        done_workload_df, on="EXP_UNIQUE_ID", how="inner"
                    )
                    detailed_metrics_df.drop_duplicates(inplace=True)

                    result_df = pd.concat(
                        [result_df, detailed_metrics_df], ignore_index=True
                    )
        return result_df
