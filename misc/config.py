import argparse
from configparser import RawConfigParser
from distutils.command.config import config
import json
import os
import pathlib
import random
import sys
from typing import List, Literal, get_args

import numpy as np

from misc.logging import init_logger


class Config:
    IGNORE_CONFIG_FILE: bool = False

    LEARNER_ML_MODEL: Literal["RF", "DT", "NB", "SVM"] = "RF"
    DATASETS_PATH: str = "~/datasets"
    N_JOBS: int = 1
    RANDOM_SEED: int = -1
    TRAIN_TEST_SPLIT: float = 0.5
    LOG_FILE: str = "console"
    RUNNING_ENVIRONMENT: Literal["local", "hpc"] = "local"

    HPC_SSH_LOGIN: str
    HPC_WS_PATH: str
    HPC_DATASET_PATH: str
    HPC_OUTPUT_PATH: str

    LOCAL_DATASET_PATH: str
    LOCAL_LOCAL_CODE_PATH: str
    LOCAL_OUTPUT_PATH: str

    EXP_TITLE: str = "tmp"
    EXP_DATASETS: List[int]
    EXP_STRATEGIES: List[int]
    EXP_RANDOM_SEEDS_START: int = 0
    EXP_RANDOM_SEEDS_END: int = 10
    EXP_RANDOM_SEEDS: List[int]
    EXP_RESULTS_FILE: str

    WORKER_INDEX: int

    def __init__(self) -> None:
        self._parse_cli_arguments()

        self._load_config_from_file(".server_access_credentials.cfg")

        # some config magic
        self.EXP_RANDOM_SEEDS = list(
            range(self.EXP_RANDOM_SEEDS_START, self.EXP_RANDOM_SEEDS_END)
        )

        if self.RUNNING_ENVIRONMENT == "local":
            self.OUTPUT_PATH = self.LOCAL_OUTPUT_PATH
        elif self.RUNNING_ENVIRONMENT == "hpc":
            self.OUTPUT_PATH = self.HPC_OUTPUT_PATH

        self.OUTPUT_PATH += "/" + self.EXP_TITLE

        # check if a config file exists which could be read in
        self._CONFIG_FILE_PATH = self.OUTPUT_PATH + "/config.json"

        if not self.IGNORE_CONFIG_FILE:
            if os.path.exists(self._CONFIG_FILE_PATH):
                with open(self._CONFIG_FILE_PATH, "r") as fp:
                    cfg_values = json.load(fp)

                for k, v in cfg_values.items():
                    if v is not None:
                        self.__setattr__(k, v)

        tmp_path = pathlib.Path(self.OUTPUT_PATH)
        tmp_path.mkdir(parents=True, exist_ok=True)

        self.EXP_RESULTS_FILE = self.OUTPUT_PATH + "/results.csv"

    def _load_config_from_file(self, path: str) -> None:
        config_parser = RawConfigParser()
        config_parser.read(path)

        for section in config_parser.sections():
            for k, v in config_parser.items(section):
                self.__setattr__(section + "_" + k.upper(), v)

    """
        Magically convert the type hints from the class attributes of this class into argparse config values
    """

    def _parse_cli_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        for k, v in Config.__annotations__.items():
            if not hasattr(Config, k):
                default = None
            else:
                default = self.__getattribute__(k)

            choices = None
            arg_type = v
            nargs = None

            if str(v).startswith("typing.Literal"):
                choices = get_args(v)
                arg_type = str
            elif str(v) == "typing.List[int]":
                nargs = "*"
                arg_type = int

            if str(v) == "<class 'bool'>":
                parser.add_argument(
                    "--" + k,
                    default=default,
                    action="store_true",
                )
            else:
                parser.add_argument(
                    "--" + k,
                    default=default,
                    type=arg_type,
                    choices=choices,
                    nargs=nargs,
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

    """
        Saves the config to a file -> can be read in later to know the details of the experiment
    """

    def save_to_file(self) -> None:
        to_save_config_values = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("HPC_")
            or not k.startswith("LOCAL_")
            or not k.startswith("_")
        }

        with open(self._CONFIG_FILE_PATH, "w") as fp:
            json.dump(to_save_config_values, fp)
