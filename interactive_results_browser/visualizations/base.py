from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, List

from typing import Any, Dict

from flask import render_template
import matplotlib.pyplot as plt
import pandas as pd
from datasets import DATASET
from interactive_results_browser.cache import memory
from resources.data_types import AL_STRATEGY, LEARNER_MODEL

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
    df: pd.DataFrame = pd.read_csv(done_workload_path)

    if limit_to_get_params:
        for k, v in exp_grid_request_params.items():
            if k in non_workload_keys:
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

    def _render_images(
        self,
        args: Dict[str, Any],
        vizualization_result_creation_function: Callable[[Any], str],
    ) -> str:
        ...

    def _load_done_workload(
        self, limit_to_get_params=True, enum_to_str=True
    ) -> pd.DataFrame:
        return _cache_load_done_workload(
            done_workload_path=self._config.DONE_WORKLOAD_PATH,
            limit_to_get_params=limit_to_get_params,
            enum_to_str=enum_to_str,
            non_workload_keys=self._NON_WORKLOAD_KEYS,
            exp_grid_request_params=self._exp_grid_request_params,
        )
