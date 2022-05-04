import argparse
import random
import sys
from configparser import RawConfigParser
from pathlib import Path
from typing import List, Literal, Union, get_args

import git
import numpy as np
import pandas as pd
import yaml

from datasets import DATASET
from misc.logging import init_logger
from ressources.data_types import AL_STRATEGY, LEARNER_MODEL


class Config:
    N_JOBS: int = 1
    RANDOM_SEED: int = -1
    LOG_FILE: str = "console"
    RUNNING_ENVIRONMENT: Literal["local", "hpc"] = "local"

    HPC_SSH_LOGIN: str
    HPC_WS_PATH: Path
    HPC_DATASETS_PATH: Path
    HPC_OUTPUT_PATH: Path

    LOCAL_DATASETS_PATH: Path
    LOCAL_CODE_PATH: Path
    LOCAL_OUTPUT_PATH: Path

    USE_EXP_YAML: str = "NOOOOO"

    EXP_TITLE: str = "tmp"
    EXP_DATASET: DATASET
    EXP_GRID_DATASET: List[DATASET]
    EXP_STRATEGY: AL_STRATEGY
    EXP_GRID_STRATEGY: List[AL_STRATEGY]
    EXP_RANDOM_SEEDS_START: int = 0
    EXP_RANDOM_SEEDS_END: int = 10
    EXP_GRID_RANDOM_SEED: List[int]
    EXP_RANDOM_SEED: int
    EXP_NUM_QUERIES: int
    EXP_GRID_NUM_QUERIES: List[int] = [0]
    EXP_BATCH_SIZE: int
    EXP_GRID_BATCH_SIZE: List[int] = [5]
    EXP_LEARNER_MODEL: LEARNER_MODEL
    EXP_GRID_LEARNER_MODEL: List[LEARNER_MODEL] = [LEARNER_MODEL.RF]
    EXP_TRAIN_TEST_BUCKET_SIZE: int
    EXP_GRID_TRAIN_TEST_BUCKET_SIZE: List[int] = list(range(0, 5))
    EXP_UNIQUE_ID: int

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
    DATASETS_TRAIN_TEST_SPLIT_APPENDIX: str = "_train_test_split.csv"
    RAW_DATASETS_PATH: Path = "_raw"  # type: ignore
    DATASETS_AMOUNT_OF_SPLITS: int = 5

    KAGGLE_DATASETS_PATH: Path = "ressources/datasets.yaml"  # type: ignore
    LOCAL_CONFIG_FILE_PATH: Path = ".server_access_credentials.cfg"  # type: ignore
    LOCAL_YAML_EXP_PATH: Path = "ressources/exp_config.yaml"  # type: ignore
    CONFIG_FILE_PATH: Path = "00_config.yaml"  # type: ignore
    WORKLOAD_FILE_PATH: Path = "01_workload.csv"  # type: ignore
    EXPERIMENT_SLURM_FILE_PATH: Path = "02_slurm.slurm"  # type: ignore
    EXPERIMENT_BASH_FILE_PATH: Path = "02_bash.sh"  # type: ignore
    EXPERIMENT_SYNC_AND_RUN_FILE_PATH: Path = "03_sync_and_run.sh"  # type: ignore
    DONE_WORKLOAD_PATH: Path = "04_done_workload.csv"  # type: ignore
    METRIC_RESULTS_PATH_APPENDIX: str = "_metric_results.csv"
    METRIC_RESULTS_FILE_PATH: Path

    def __init__(self) -> None:
        self._parse_cli_arguments()
        self._load_server_setup_from_file(Path(self.LOCAL_CONFIG_FILE_PATH))

        self._pathes_magic()

        # check if we have a yaml defined experiment
        if self.USE_EXP_YAML != "NOOOOO":
            # yes, we have -> overwrite everything, except for the stuff which was explicitly defined
            self._load_exp_yaml()

        self.EXP_GRID_RANDOM_SEED = list(
            range(self.EXP_RANDOM_SEEDS_START, self.EXP_RANDOM_SEEDS_END)
        )

        if self.RANDOM_SEED != -1 and self.RANDOM_SEED != -2:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

        if self.WORKER_INDEX is not None:
            self.load_workload()

            # BUG worker_index ist bei wiederholten runs nicht eindeutig um das ergebnis abzuspeichern! --> write a new field called "original workload id into workload.csv" -> and if that file gets recreated -> use the old workload id for that workload!
            # BUG workload wird einmal generiert -> und enthält die orgiinal ids
            # wenn das jetzt nochmal gemacht werden soll -> wir lesen zuerst das originale mit den original ids ein, und entfernen dann alle die, wo es die ids bereits gibt, und speichern es dann erneut ab --> dann sparen wir uns auch das param_grid nochmal neu zu berechnen!
            # magically create the output path
            self.METRIC_RESULTS_FILE_PATH = (
                self.OUTPUT_PATH
                / self.EXP_DATASET.name
                / str(str(self.EXP_UNIQUE_ID) + self.METRIC_RESULTS_PATH_APPENDIX)
            )
            self.METRIC_RESULTS_FILE_PATH.parent.mkdir(exist_ok=True)

    def _pathes_magic(self) -> None:
        if self.RUNNING_ENVIRONMENT == "local":
            self.OUTPUT_PATH = Path(self.LOCAL_OUTPUT_PATH)
            self.DATASETS_PATH = Path(self.LOCAL_DATASETS_PATH)
        elif self.RUNNING_ENVIRONMENT == "hpc":
            self.OUTPUT_PATH = Path(self.HPC_OUTPUT_PATH)
            self.DATASETS_PATH = Path(self.HPC_DATASETS_PATH)

        if self.USE_EXP_YAML != "NOOOOO":
            self.EXP_TITLE = self.USE_EXP_YAML

        self.OUTPUT_PATH = self.OUTPUT_PATH / self.EXP_TITLE

        # check if a config file exists which could be read in
        self.CONFIG_FILE_PATH = self.OUTPUT_PATH / self.CONFIG_FILE_PATH

        self.LOCAL_CONFIG_FILE_PATH = Path(self.LOCAL_CONFIG_FILE_PATH)
        self.CONFIG_FILE_PATH = Path(self.CONFIG_FILE_PATH)
        self.WORKLOAD_FILE_PATH = self.OUTPUT_PATH / self.WORKLOAD_FILE_PATH
        self.EXPERIMENT_SLURM_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_FILE_PATH
        )
        self.EXPERIMENT_BASH_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_BASH_FILE_PATH
        )

        self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH
        )

        self.RAW_DATASETS_PATH = self.DATASETS_PATH / self.RAW_DATASETS_PATH

        self.KAGGLE_DATASETS_PATH = Path(self.KAGGLE_DATASETS_PATH)

        self.DONE_WORKLOAD_PATH = self.OUTPUT_PATH / self.DONE_WORKLOAD_PATH

        self.OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    def _return_list_of_explicitly_defined_cli_args(self) -> List[str]:
        explicitly_defined_arguments: List[str] = []
        for arg in sys.argv:
            if arg.startswith("--"):
                explicitly_defined_arguments.append(arg[2:])
        return explicitly_defined_arguments

    def _load_server_setup_from_file(self, config_path: Path) -> None:
        config_parser = RawConfigParser()
        config_parser.read(config_path)

        # check, which arguments have been specified in the args list
        explicitly_defined_arguments = (
            self._return_list_of_explicitly_defined_cli_args()
        )

        for section in config_parser.sections():
            for k, v in config_parser.items(section):
                if section + "_" + k.upper() in explicitly_defined_arguments:
                    # we do not overwrite our config with arguments which have been specified as CLI arguments
                    continue
                self.__setattr__(section + "_" + k.upper(), v)

    def _load_exp_yaml(self) -> None:
        yaml_config_params = yaml.safe_load(self.LOCAL_YAML_EXP_PATH.read_bytes())

        yaml_config_params = yaml_config_params[self.USE_EXP_YAML]

        self.EXP_TITLE = self.USE_EXP_YAML

        explicitly_defined_cli_args = self._return_list_of_explicitly_defined_cli_args()

        for k, v in yaml_config_params.items():
            if k in explicitly_defined_cli_args:
                continue

            # convert str/ints to enum data types first
            if k == "EXP_GRID_STRATEGY":
                if type(v[0]) == int:
                    v = [AL_STRATEGY(x) for x in v]
                else:
                    v = [AL_STRATEGY[x] for x in v]
            elif k == "EXP_GRID_DATASET":
                if type(v[0]) == int:
                    v = [DATASET(x) for x in v]
                else:
                    v = [DATASET[x] for x in v]
            elif k == "EXP_GRID_LEARNER_MODEL":
                if type(v[0]) == int:
                    v = [LEARNER_MODEL(x) for x in v]
                else:
                    v = [LEARNER_MODEL[x] for x in v]

            self.__setattr__(k, v)

    def load_workload(self) -> None:
        workload_df = pd.read_csv(
            self.WORKLOAD_FILE_PATH,
            header=0,
            index_col=None,
            skiprows=lambda x: x not in [0, self.WORKER_INDEX + 1],
        )
        workload = workload_df.iloc[0].to_dict()
        for k, v in workload.items():
            print(f"{k}\t\t\t{v}")
            # convert str/ints to enum data types first
            if k == "EXP_STRATEGY":
                v = AL_STRATEGY(int(v))
            elif k == "EXP_DATASET":
                v = DATASET(int(v))
            elif k == "EXP_LEARNER_MODEL":
                v = LEARNER_MODEL(int(v))

            if str(self.__annotations__[k]).endswith("int]"):
                v = int(v)
            self.__setattr__(k, v)

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
            elif (
                str(v) == "typing.List[int]" or str(v) == "typing.Union[typing.List[int"
            ):
                nargs = "*"
                arg_type = int
            # enum classes:
            elif (
                str(v).startswith("typing.List[misc")
                or str(v) == "typing.List[datasets.DATASET]"
            ):
                full_str = str(v).split("[")[1][:-1].split(".")
                module_str = ".".join(full_str[:-1])
                class_str = full_str[-1]
                v_class = getattr(sys.modules[module_str], class_str)

                # allow all possible integer values from the enum classes
                choices = [e.value for e in v_class]  # type: ignore
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
                    nargs=nargs,  # type: ignore
                )

        config: argparse.Namespace = parser.parse_args()

        for k, v in config.__dict__.items():
            self.__setattr__(k, v)

        # if len(sys.argv[:-1]) == 0:
        #    parser.print_help()

        init_logger(self.LOG_FILE)

    """
        Saves the config to a file -> can be read in later to know the details of which the experiment was run with
    """

    def save_to_file(self) -> None:
        to_save_config_values = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("HPC_")
            and not k.startswith("LOCAL_")
            and not k.startswith("_")
        }

        to_save_config_values["GIT_COMMIT_HASH"] = git.Repo(  # type: ignore
            search_parent_directories=True
        ).head.object.hexsha

        self.CONFIG_FILE_PATH.write_text(yaml.dump(to_save_config_values))
