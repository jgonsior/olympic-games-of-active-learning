import argparse
from configparser import RawConfigParser
from distutils.command.config import config
from enum import Enum, IntEnum, auto
import json
import os
import random
import sys
from typing import List, Literal, NewType, Union, get_args
from pathlib import Path
import git

import numpy as np
from sklearn import naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import explained_variance_score
from sklearn.tree import DecisionTreeClassifier

from misc.logging import init_logger


class Dataset(IntEnum):
    DWTC = 1
    FLAG = 2
    IRIS = 3
    WINE = 4
    GERMAN = 5


class Strategy(IntEnum):
    ALIPY_RANDOM = 1
    ALIPY_UNCERTAINTY_LC = 2
    ALIPY_UNCERTAINTY_ENT = 3
    ALIPY_UNCERTAINTY_MM = 4
    ALIPY_UNCERTAINTY_QUIRE = 5


class SKLEARN_ML_MODELS(IntEnum):
    RF = 1
    DT = 2
    NB = 3
    SVM = 4


class Config:
    IGNORE_CONFIG_FILE: bool = False

    LEARNER_ML_MODEL: SKLEARN_ML_MODELS = SKLEARN_ML_MODELS.RF
    N_JOBS: int = 1
    RANDOM_SEED: int = -1
    TRAIN_TEST_SPLIT: float = 0.5
    LOG_FILE: str = "console"
    RUNNING_ENVIRONMENT: Literal["local", "hpc"] = "local"

    HPC_SSH_LOGIN: str
    HPC_WS_PATH: Path
    HPC_DATASETS_PATH: Path
    HPC_OUTPUT_PATH: Path

    LOCAL_DATASETS_PATH: Path
    LOCAL_CODE_PATH: Path
    LOCAL_OUTPUT_PATH: Path

    EXP_TITLE: str = "tmp"
    EXP_DATASETS: List[Dataset]
    EXP_STRATEGIES: List[Strategy]
    EXP_RANDOM_SEEDS_START: int = 0
    EXP_RANDOM_SEEDS_END: int = 10
    EXP_RANDOM_SEEDS: List[int]

    WORKER_INDEX: int

    SLURM_TIME_LIMIT: str = "1:59:59"
    SLURM_NR_THREADS: int = 1
    SLURM_MEMORY: int = 1875
    HPC_SLURM_MAIL: str
    HPC_SLURM_PROJECT: str
    SLURM_OFFSET: int = 0
    SLURM_ITERATIONS_PER_BATCH: int = 10

    BASH_PARALLEL_RUNNERS: int = 10

    DATASETS_PATH: Path
    LOCAL_CONFIG_FILE_PATH: Path = ".server_access_credentials.cfg"  # type: ignore
    CONFIG_FILE_PATH: Path = "00_config.json"  # type: ignore
    WORKLOAD_FILE_PATH: Path = "01_workload.csv"  # type: ignore
    EXPERIMENT_SLURM_FILE_PATH: Path = "02_slurm.slurm"  # type: ignore
    EXPERIMENT_BASH_FILE_PATH: Path = "02_bash.sh"  # type: ignore
    EXPERIMENT_SYNC_AND_RUN_FILE_PATH: Path = "03_sync_and_run.sh"  # type: ignore
    RESULTS_FILE_PATH: Path = "04_results.csv"  # type: ignore

    def __init__(self) -> None:
        self._parse_cli_arguments()
        self._load_config_from_file(Path(self.LOCAL_CONFIG_FILE_PATH))

        # some config magic
        self.EXP_RANDOM_SEEDS = list(
            range(self.EXP_RANDOM_SEEDS_START, self.EXP_RANDOM_SEEDS_END)
        )

        if self.RUNNING_ENVIRONMENT == "local":
            self.OUTPUT_PATH = Path(self.LOCAL_OUTPUT_PATH)
        elif self.RUNNING_ENVIRONMENT == "hpc":
            self.OUTPUT_PATH = Path(self.HPC_OUTPUT_PATH)

        self.OUTPUT_PATH = self.OUTPUT_PATH / self.EXP_TITLE

        # check if a config file exists which could be read in
        self.CONFIG_FILE_PATH = self.OUTPUT_PATH / self.CONFIG_FILE_PATH

        self._create_pathes()

        if not self.IGNORE_CONFIG_FILE:
            if os.path.exists(self.CONFIG_FILE_PATH):
                cfg_values = json.loads(self.CONFIG_FILE_PATH.read_text())

                for k, v in cfg_values.items():
                    if v is not None:
                        self.__setattr__(k, v)

        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    def _create_pathes(self) -> None:
        self.LOCAL_CONFIG_FILE_PATH = Path(self.LOCAL_CONFIG_FILE_PATH)
        self.CONFIG_FILE_PATH = Path(self.CONFIG_FILE_PATH)
        self.WORKLOAD_FILE_PATH = self.OUTPUT_PATH / self.WORKLOAD_FILE_PATH
        self.EXPERIMENT_SLURM_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_FILE_PATH
        )
        self.EXPERIMENT_BASH_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_BASH_FILE_PATH
        )
        self.RESULTS_FILE_PATH = self.OUTPUT_PATH / self.RESULTS_FILE_PATH
        self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH
        )

    def _load_config_from_file(self, config_path: Path) -> None:
        config_parser = RawConfigParser()
        config_parser.read(config_path)

        # check, which arguments have been specified in the args list
        # TODO
        explicitly_defined_arguments: List[str] = []
        for arg in sys.argv:
            if arg.startswith("--"):
                explicitly_defined_arguments.append(arg[2:])

        for section in config_parser.sections():
            for k, v in config_parser.items(section):
                if section + "_" + k.upper() in explicitly_defined_arguments:
                    # we do not overwrite our config with arguments which have been specified as CLI arguments
                    continue
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
            elif str(v).startswith("typing.List[misc.config."):
                full_str = str(v).split("[")[1][:-1].split(".")
                module_str = ".".join(full_str[:-1])
                class_str = full_str[-1]
                v_class = getattr(sys.modules[module_str], class_str)

                # allow all possible integer values from the enum classes
                choices = [e.value for e in v_class]
                nargs = "*"
                arg_type = int
            elif str(v) == "typing.Union[str, pathlib.Path]":
                arg_type = str

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
            and not k.startswith("LOCAL_")
            and not k.startswith("_")
        }

        to_save_config_values["GIT_COMMIT_HASH"] = git.Repo(
            search_parent_directories=True
        ).head.object.hexsha

        def _default_json(t):
            return f"{t}"

        self.CONFIG_FILE_PATH.write_text(
            json.dumps(to_save_config_values, default=_default_json)
        )
