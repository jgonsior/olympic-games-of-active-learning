import argparse
import os
import random
import sys
from configparser import RawConfigParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import git
import numpy as np
import yaml

from datasets import DATASET
from misc.logging import init_logger, log_it
from resources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
    _import_compiled_libact_strategies,
)


class Config:
    N_JOBS: int = 1
    RANDOM_SEED: int = -1
    LOG_FILE: str = "console"
    RUNNING_ENVIRONMENT: Literal["local", "hpc"] = "local"

    HPC_SSH_LOGIN: str
    HPC_WS_PATH: Path
    HPC_DATASETS_PATH: Path
    HPC_OUTPUT_PATH: Path
    HPC_CODE_PATH: Path

    LOCAL_DATASETS_PATH: Path
    LOCAL_CODE_PATH: Path
    LOCAL_OUTPUT_PATH: Path

    INCLUDE_RESULTS_FROM: List[str]

    EXP_TITLE: str = "test_exp_2"
    EXP_DATASET: DATASET
    EXP_GRID_DATASET: List[DATASET]
    EXP_STRATEGY: AL_STRATEGY
    EXP_GRID_STRATEGY: List[AL_STRATEGY]
    EXP_GRID_RANDOM_SEEDS_START: int = 0
    EXP_GRID_RANDOM_SEEDS_END: int
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
    SLURM_JOBS_PR_THREAD = 10
    HPC_SLURM_MAIL: str
    HPC_SLURM_PROJECT: str
    SLURM_OFFSET: int = 0
    SLURM_ITERATIONS_PER_BATCH: int = 100

    BASH_PARALLEL_RUNNERS: int = 10

    DATASETS_PATH: Path
    DATASETS_TRAIN_TEST_SPLIT_APPENDIX: str = "_train_test_split.csv"
    RAW_DATASETS_PATH: Path = "_raw"  # type: ignore
    DATASETS_AMOUNT_OF_SPLITS: int = 5
    DATASETS_TEST_SIZE_PERCENTAGE: float = 0.4

    KAGGLE_DATASETS_PATH: Path = "resources/datasets.yaml"  # type: ignore
    LOCAL_CONFIG_FILE_PATH: Path = ".server_access_credentials.cfg"  # type: ignore
    LOCAL_YAML_EXP_PATH: Path = "resources/exp_config.yaml"  # type: ignore
    CONFIG_FILE_PATH: Path = "00_config.yaml"  # type: ignore
    WORKLOAD_FILE_PATH: Path = "01_workload.csv"  # type: ignore
    EXPERIMENT_SLURM_FILE_PATH: Path = "02_slurm.slurm"  # type: ignore
    EXPERIMENT_SLURM_CHAIN_JOB: Path = "02b_chain_job.sh"  # type: ignore
    EXPERIMENT_SLURM_TAR_PATH: Path = "03_tar.slurm"  # type: ignore
    EXPERIMENT_BASH_FILE_PATH: Path = "02_bash.sh"  # type: ignore
    EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH: Path = "02b_run_bash_parallel.py"  # type: ignore
    EXPERIMENT_SYNC_AND_RUN_FILE_PATH: Path = "04_sync_and_run.sh"  # type: ignore
    DONE_WORKLOAD_PATH: Path = "05_done_workload.csv"  # type: ignore
    METRIC_RESULTS_PATH_APPENDIX: str = "_metric_results.csv.gz"
    EXP_RESULT_ZIP_PATH_PREFIX: Path
    EXP_RESULT_ZIP_PATH: Path = ".tar.gz"  # type: ignore
    EXP_RESULT_EXTRACTED_ZIP_PATH: Path
    METRIC_RESULTS_FILE_PATH: Path

    DONE_WORKLOAD_FILE: Path
    RESULTS_PATH: Path
    HTML_STATUS_PATH: Path = "06_status.html"  # type:ignore

    def __init__(self, no_cli_args: Optional[Dict[str, Any]] = None) -> None:
        if no_cli_args is not None:
            self._parse_non_cli_arguments(no_cli_args)
        else:
            self._parse_cli_arguments()
        self._setup_everything()

    def _parse_non_cli_arguments(self, no_cli_args: Dict[str, Any]) -> None:
        for k, v in no_cli_args.items():
            self.__setattr__(k, v)

    def _setup_everything(self):
        self._load_server_setup_from_file(Path(self.LOCAL_CONFIG_FILE_PATH))

        if not Path(self.HPC_CODE_PATH).exists():
            self.RUNNING_ENVIRONMENT = "local"
            _import_compiled_libact_strategies()
        else:
            self.RUNNING_ENVIRONMENT = "hpc"

        self._pathes_magic()

        # load yaml and overwrite everything, except for the stuff which was explicitly defined
        self._load_exp_yaml()

        self.EXP_GRID_RANDOM_SEED = list(
            range(self.EXP_GRID_RANDOM_SEEDS_START, self.EXP_GRID_RANDOM_SEEDS_END)
        )

        if self.RANDOM_SEED != -1 and self.RANDOM_SEED != -2:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

        if self.WORKER_INDEX is not None:
            self.load_workload()

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

        self.EXP_RESULT_ZIP_PATH_PREFIX = Path(
            str(self.HPC_WS_PATH)[1:] + "exp_results/" + self.EXP_TITLE
        )
        self.EXP_RESULT_ZIP_PATH = Path(
            str(self.OUTPUT_PATH) + str(self.EXP_RESULT_ZIP_PATH)
        )

        self.EXP_RESULT_EXTRACTED_ZIP_PATH = (
            self.OUTPUT_PATH / self.EXP_RESULT_ZIP_PATH_PREFIX
        )

        self.OUTPUT_PATH = self.OUTPUT_PATH / self.EXP_TITLE

        # check if a config file exists which could be read in
        self.CONFIG_FILE_PATH = self.OUTPUT_PATH / self.CONFIG_FILE_PATH

        self.LOCAL_YAML_EXP_PATH = Path(self.LOCAL_YAML_EXP_PATH)

        self.LOCAL_CONFIG_FILE_PATH = Path(self.LOCAL_CONFIG_FILE_PATH)
        self.CONFIG_FILE_PATH = Path(self.CONFIG_FILE_PATH)
        self.WORKLOAD_FILE_PATH = self.OUTPUT_PATH / self.WORKLOAD_FILE_PATH
        self.EXPERIMENT_SLURM_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_FILE_PATH
        )
        self.EXPERIMENT_SLURM_CHAIN_JOB = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_CHAIN_JOB
        )
        self.EXPERIMENT_SLURM_TAR_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_TAR_PATH
        )
        self.EXPERIMENT_BASH_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_BASH_FILE_PATH
        )
        self.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH
        )

        self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SYNC_AND_RUN_FILE_PATH
        )

        self.HTML_STATUS_PATH = self.OUTPUT_PATH / self.HTML_STATUS_PATH

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

        yaml_config_params = yaml_config_params[self.EXP_TITLE]

        explicitly_defined_cli_args = self._return_list_of_explicitly_defined_cli_args()

        for k, v in yaml_config_params.items():
            if k in explicitly_defined_cli_args:
                continue

            # convert str/ints to enum data types first
            if k == "EXP_GRID_STRATEGY":
                # parse args
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
        import pandas as pd

        workload_df = pd.read_csv(
            self.WORKLOAD_FILE_PATH,
            header=0,
            index_col=None,
            skiprows=lambda x: x not in [0, self.WORKER_INDEX + 1],
        )
        workload = workload_df.iloc[0].to_dict()
        for k, v in workload.items():
            # log_it(f"{k}\t\t\t{str(v)}")
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

        for k in workload.keys():
            log_it(f"{k}\t\t\t{str(self.__getattribute__(k))}")

        np.random.seed(self.EXP_RANDOM_SEED)
        random.seed(self.EXP_RANDOM_SEED)

        self._original_workload = workload

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
            elif str(v) == "typing.List[str]":
                nargs = "*"
                arg_type = str
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
