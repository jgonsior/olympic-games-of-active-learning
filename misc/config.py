# find a way, to specify the config parameters firs twith a datatype and a range, and default values, so that it can be used in multiple files?
# --> reusable configs/same configs for 01_ and 02_!


import argparse
from configparser import RawConfigParser
import random
import sys
from typing import Any, Dict, List, Tuple
import datetime
import threading

import numpy as np

from misc.logging import init_logger


class Config:
    _config_values: Dict[str, Any] = {}

    def __init__(self) -> None:
        self._parse_cli_arguments()
        self._load_config_from_file(".server_access_credentials.cfg")
        pass

    def _load_config_from_file(self, path: str) -> None:
        config_parser = RawConfigParser()
        config_parser.read(path)
        print(config_parser)

        self._config_values = {**self._config_values, **config_parser}

    def _parse_cli_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--DATASETS_PATH", default="~/datasets")
        parser.add_argument(
            "--CLASSIFIER",
            default="RF",
            help="Supported types: RF, DTree, NB, SVM, Linear",
        )
        parser.add_argument("--N_JOBS", type=int, default=-1)
        parser.add_argument(
            "--RANDOM_SEED", type=int, default=42, help="-1 Enables true Randomness"
        )
        parser.add_argument("--TEST_FRACTION", type=float, default=0.5)
        parser.add_argument("--LOG_FILE", type=str, default="log.txt")

        config: argparse.Namespace = parser.parse_args()

        if len(sys.argv[:-1]) == 0:
            parser.print_help()

        self._config_values = {**self._config_values, **config.__dict__}

        if self.RANDOM_SEED != -1 and self.RANDOM_SEED != -2:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

        init_logger(self.LOG_FILE)

    def __getattr__(self, name: str) -> Any:
        if name not in self.__dict__["_config_values"]:
            print(self.__dict__["_config_values"].keys())
            print("Config Key not existent")
        return self.__dict__["_config_values"][name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if not "_config_values" in self.__dict__:
            self.__dict__["_config_values"] = {}
        if __name == "_config_values":
            self.__dict__[__name] = __value
        else:
            self.__dict__["_config_values"][__name] = __value
