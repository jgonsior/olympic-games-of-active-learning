# find a way, to specify the config parameters firs twith a datatype and a range, and default values, so that it can be used in multiple files?
# --> reusable configs/same configs for 01_ and 02_!


import argparse
from configparser import RawConfigParser
import random
from secrets import choice
import sys
from typing import Any, Dict, List, Literal, Tuple, get_args

import numpy as np

from misc.logging import init_logger


class Config:
    LEARNER_ML_MODEL: Literal["RF", "DT", "NB", "SVM"] = "RF"
    DATASETS_PATH: str = "~/datasets"
    N_JOBS: int = 1
    RANDOM_SEED: int = -1
    TRAIN_TEST_SPLIT: float = 0.5
    LOG_FILE: str = "console"

    HPC_SSH_LOGIN: str
    HPC_WS_PATH: str
    HPC_DATASET_PATH: str
    HPC_OUTPUT_PATH: str

    LOCAL_DATASET_PATH: str
    LOCAL_LOCAL_CODE_PATH: str
    LOCAL_OUTPUT_PATH: str

    def __init__(self) -> None:
        self._parse_cli_arguments()
        self._load_config_from_file(".server_access_credentials.cfg")

    def _load_config_from_file(self, path: str) -> None:
        config_parser = RawConfigParser()
        config_parser.read(path)

        for section in config_parser.sections():
            for k, v in config_parser.items(section):
                self.__setattr__(section + "_" + k.upper(), v)

    def _parse_cli_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        for k, v in Config.__annotations__.items():
            if not hasattr(Config, k):
                default = None
            else:
                default = self.__getattribute__(k)

            choices = None
            arg_type = v
            if str(v).startswith("typing.Literal"):
                choices = get_args(v)
                arg_type = str

            parser.add_argument(
                "--" + k,
                default=default,
                type=arg_type,
                choices=choices,
            )

        config: argparse.Namespace = parser.parse_args()

        for k, v in config.__dict__.items():
            self.__setattr__(k, v)

        if len(sys.argv[:-1]) == 0:
            parser.print_help()

        if self.RANDOM_SEED != -1 and self.RANDOM_SEED != -2:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

        init_logger(self.LOG_FILE)

    # def __getitem__(self, key):
    #    return getattr(self, key)

    # def __setitem__(self, key, value):
    #    setattr(self, key, value)
