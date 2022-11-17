from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, List

from typing import Any, Dict

from flask import render_template
import matplotlib.pyplot as plt
import pandas as pd
from datasets import DATASET
from interactive_results_browser.cache import cache
from resources.data_types import AL_STRATEGY, LEARNER_MODEL

if TYPE_CHECKING:
    from misc.config import Config


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

    def _cache_result(
        self,
        args: Dict[str, Any],
        vizualization_result_creation_function: Callable[[Any], str],
    ) -> str:
        # first check if args_vizualization_result_creation_function exists in cache
        # if so -> return the cached vizualization, do not run the cache
        # if not -> create vizualization, and we're happy!

        cache_key = vizualization_result_creation_function.__name__ + "_" + str(args)

        if cache.has(cache_key):
            vizualization_result = cache.get(cache_key)
            print(f"Cache found: {cache_key}")
        else:
            print(f"Cache not found: {cache_key}")
            vizualization_result = vizualization_result_creation_function(*args)

            # convert matplotlib object into html png
            cache.set(cache_key, vizualization_result)

        return vizualization_result

    def _load_done_workload(
        self, limit_to_get_params=True, enum_to_str=True
    ) -> pd.DataFrame:
        def _cache_load_done_workload(limit_to_get_params, enum_to_str):
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
                df["EXP_STRATEGY"] = df["EXP_STRATEGY"].apply(
                    lambda x: AL_STRATEGY(x).name
                )
            return df

        return self._cache_result(
            args={
                "limit_to_get_params": limit_to_get_params,
                "enum_to_str": enum_to_str,
            },
            vizualization_result_creation_function=_cache_load_done_workload,
        )
