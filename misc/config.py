import argparse
import random
import sys
from configparser import RawConfigParser
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, get_args
from aenum import extend_enum

import git
import numpy as np
import yaml

from datasets import DATASET
from metrics.base_metric import Base_Metric
from misc.logging import init_logger, log_it
from resources.data_types import (
    AL_STRATEGY,
    COMPUTED_METRIC,
    SAMPLES_CATEGORIZER,
    LEARNER_MODEL,
)


class Config:
    N_JOBS: int = 1
    RANDOM_SEED: int = 1312
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

    AMOUNT_OF_START_POINTS_TO_GENERATE: int = 10000

    EXP_TITLE: str = "all_strategies_all_datasets_single_random_seed"
    EXP_DATASET: DATASET
    EXP_GRID_DATASET: List[DATASET]
    EXP_STRATEGY: AL_STRATEGY
    EXP_GRID_STRATEGY: List[AL_STRATEGY]
    EXP_RANDOM_SEED: int
    EXP_GRID_RANDOM_SEED: List[int]
    EXP_START_POINT: int
    EXP_GRID_START_POINT: List[int]
    EXP_NUM_QUERIES: int
    EXP_GRID_NUM_QUERIES: List[int] = [0]
    EXP_BATCH_SIZE: int
    EXP_GRID_BATCH_SIZE: List[int] = [5]
    EXP_LEARNER_MODEL: LEARNER_MODEL
    EXP_GRID_LEARNER_MODEL: List[LEARNER_MODEL] = [LEARNER_MODEL.RF]
    EXP_TRAIN_TEST_BUCKET_SIZE: int
    EXP_GRID_TRAIN_TEST_BUCKET_SIZE: List[int] = list(range(0, 5))
    EXP_UNIQUE_ID: int
    EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT: int = 300  # 600  10 minutes

    WORKER_INDEX: int

    SLURM_TIME_LIMIT: str = "1:59:59"
    SLURM_NR_THREADS: int = 1
    SLURM_MEMORY: int = 2210
    SLURM_JOBS_PR_THREAD = 10
    HPC_SLURM_MAIL: str
    HPC_SLURM_PROJECT: str
    SLURM_OFFSET: int = 0
    SLURM_ITERATIONS_PER_BATCH: int = 100

    BASH_PARALLEL_RUNNERS: int = 10

    DATASETS_PATH: Path
    DATASETS_TRAIN_TEST_SPLIT_APPENDIX: str = "_split.csv"
    RAW_DATASETS_PATH: Path = "_raw"  # type: ignore
    DATASETS_AMOUNT_OF_SPLITS: int = 5
    DATASETS_TEST_SIZE_PERCENTAGE: float = 0.4
    DATASETS_COMPUTE_DISTANCES: bool = True
    DATASETS_DISTANCES_APPENDIX: str = "_distances.csv.gzip"

    KAGGLE_DATASETS_YAML_CONFIG_PATH: Path = "resources/kaggle_datasets.yaml"  # type: ignore
    OPENML_DATASETS_YAML_CONFIG_PATH: Path = "resources/openml_datasets.yaml"  # type: ignore
    LOCAL_DATASETS_YAML_CONFIG_PATH: Path = "resources/local_datasets.yaml"  # type: ignore
    LOCAL_CONFIG_FILE_PATH: Path = ".server_access_credentials.cfg"  # type: ignore
    LOCAL_YAML_EXP_PATH: Path = "resources/exp_config.yaml"  # type: ignore
    CONFIG_FILE_PATH: Path = "00_config.yaml"  # type: ignore
    WORKLOAD_FILE_PATH: Path = "01_workload.csv"  # type: ignore
    NON_HPC_WORKLOAD_FILE_PATH: Path = "01_non_hpc_workload.csv"  # type: ignore
    SEPARATE_HPC_LOCAL_WORKLOAD: bool = False
    EXPERIMENT_INSTALL_SLURM_DEP_PATH = "0x_install_deps.slurm"
    EXPERIMENT_UPDATE_SLURM_DEP_PATH = "0x_update_deps.slurm"
    EXPERIMENT_SLURM_FILE_PATH: Path = "02_slurm.slurm"  # type: ignore
    EXPERIMENT_SLURM_CHAIN_JOB: Path = "02b_chain_job.sh"  # type: ignore
    EXPERIMENT_SLURM_GZIP_RESULTS_PATH = "02c_gzip_results.sh.slurm"  # type: ignore
    EXPERIMENT_SLURM_TAR_PATH: Path = "03_tar.slurm"  # type: ignore
    EXPERIMENT_BASH_FILE_PATH: Path = "02_bash.sh"  # type: ignore
    EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH: Path = "02b_run_bash_parallel.py"  # type: ignore
    EXPERIMENT_SYNC_AND_RUN_FILE_PATH: Path = "04_sync_and_run.sh"  # type: ignore
    OVERALL_DONE_WORKLOAD_PATH: Path = "05_done_workload.csv"  # type: ignore
    OVERALL_FAILED_WORKLOAD_PATH: Path = "05_failed_workloads.csv"  # type: ignore
    OVERALL_STARTED_OOM_WORKLOAD_PATH: Path = "05_started_oom_workloads.csv"  # type: ignore
    WRONG_CSV_FILES_PATH: Path = "05_wrong_csv_files.csv"  # type: ignore
    EXP_ID_METRIC_CSV_FOLDER_PATH: Path = "metrics"  # type: ignore
    DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH: Path = "_dataset_dependent_random_ramp_plateau_threshold.csv"  # type: ignore

    EXP_RESULT_ZIP_PATH_PREFIX: Path
    EXP_RESULT_ZIP_PATH: Path = ".tar.gz"  # type: ignore
    EXP_RESULT_EXTRACTED_ZIP_PATH: Path
    METRIC_RESULTS_FOLDER: Path

    DENSE_WORKLOAD_PATH: Path = "06_dense_workload.csv"  # type: ignore
    MISSING_EXP_IDS_IN_METRIC_FILES: Path = "07_missing_exp_ids.csv"  # type: ignore
    BROKEN_CSV_FILE_PATH: Path = "07_broken_csv_file_found.csv"  # type: ignore

    RERUN_FAILED_WORKLOADS: bool = False
    RECALCULATE_UPDATED_EXP_GRID: bool = False
    OVERWRITE_EXISTING_METRIC_FILES: bool = False

    RESULTS_PATH: Path

    METRICS: List[Base_Metric]
    COMPUTED_METRICS: List[COMPUTED_METRIC]
    SAMPLES_CATEGORIZER: List[SAMPLES_CATEGORIZER]
    CORRELATION_TS_PATH: Path = "_TS"  # ignore

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

        self._pathes_magic()

        # load yaml and overwrite everything, except for the stuff which was explicitly defined
        self._load_exp_yaml()

        if self.RANDOM_SEED != -1 and self.RANDOM_SEED != -2:
            np.random.seed(self.RANDOM_SEED)
            random.seed(self.RANDOM_SEED)

        if self.WORKER_INDEX is not None:
            self.load_workload()

            self.METRIC_RESULTS_FOLDER = (
                self.OUTPUT_PATH / self.EXP_STRATEGY.name / self.EXP_DATASET.name
            )
            self.METRIC_RESULTS_FOLDER.mkdir(exist_ok=True, parents=True)

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
        self.NON_HPC_WORKLOAD_FILE_PATH = (
            self.OUTPUT_PATH / self.NON_HPC_WORKLOAD_FILE_PATH
        )
        self.EXPERIMENT_SLURM_FILE_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_FILE_PATH
        )
        self.EXPERIMENT_UPDATE_SLURM_DEP_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_UPDATE_SLURM_DEP_PATH
        )
        self.EXPERIMENT_INSTALL_SLURM_DEP_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_INSTALL_SLURM_DEP_PATH
        )
        self.EXPERIMENT_SLURM_GZIP_RESULTS_PATH = (
            self.OUTPUT_PATH / self.EXPERIMENT_SLURM_GZIP_RESULTS_PATH
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

        self.RAW_DATASETS_PATH = self.DATASETS_PATH / self.RAW_DATASETS_PATH

        self.KAGGLE_DATASETS_YAML_CONFIG_PATH = Path(
            self.KAGGLE_DATASETS_YAML_CONFIG_PATH
        )
        self.OPENML_DATASETS_YAML_CONFIG_PATH = Path(
            self.OPENML_DATASETS_YAML_CONFIG_PATH
        )
        self.LOCAL_DATASETS_YAML_CONFIG_PATH = Path(
            self.LOCAL_DATASETS_YAML_CONFIG_PATH
        )

        self.OVERALL_DONE_WORKLOAD_PATH = (
            self.OUTPUT_PATH / self.OVERALL_DONE_WORKLOAD_PATH
        )

        self.DENSE_WORKLOAD_PATH = self.OUTPUT_PATH / self.DENSE_WORKLOAD_PATH
        self.MISSING_EXP_IDS_IN_METRIC_FILES = (
            self.OUTPUT_PATH / self.MISSING_EXP_IDS_IN_METRIC_FILES
        )
        self.BROKEN_CSV_FILE_PATH = self.OUTPUT_PATH / self.BROKEN_CSV_FILE_PATH

        self.OVERALL_FAILED_WORKLOAD_PATH = (
            self.OUTPUT_PATH / self.OVERALL_FAILED_WORKLOAD_PATH
        )
        self.OVERALL_STARTED_OOM_WORKLOAD_PATH = (
            self.OUTPUT_PATH / self.OVERALL_STARTED_OOM_WORKLOAD_PATH
        )

        self.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH = (
            self.OUTPUT_PATH / self.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH
        )

        self.WRONG_CSV_FILES_PATH = self.OUTPUT_PATH / self.WRONG_CSV_FILES_PATH
        self.EXP_ID_METRIC_CSV_FOLDER_PATH = (
            self.OUTPUT_PATH / self.EXP_ID_METRIC_CSV_FOLDER_PATH
        )

        self.CORRELATION_TS_PATH = self.OUTPUT_PATH / self.CORRELATION_TS_PATH

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

        # extract int ranges for datatypes who could potentially contain lists
        for k, v in yaml_config_params.items():
            v = str(v)

            # what to do with exp_datasets containing -?
            allowed_symbols = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-"]
            only_allowed = True

            for vvv in v[2:-2]:
                if vvv not in allowed_symbols:
                    only_allowed = False
            if v.startswith("['") and v.endswith("']") and "-" in v and only_allowed:
                v = v[2:-2]
                v = v.split("-")
                yaml_config_params[k] = [iii for iii in range(int(v[0]), int(v[1]) + 1)]

        # check if dataset args ar not in the DATASET enmus
        # if they are not -> add them to it
        if self.LOCAL_DATASETS_YAML_CONFIG_PATH.exists():
            local_datasets_yaml_config = yaml.safe_load(
                self.LOCAL_DATASETS_YAML_CONFIG_PATH.read_text()
            )
            if local_datasets_yaml_config != None:
                for k in local_datasets_yaml_config.keys():
                    if k not in [d.name for d in DATASET]:
                        extend_enum(
                            DATASET, k, local_datasets_yaml_config[k]["enum_id"]
                        )

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
            # convert str/ints to enum data types first
            if k == "EXP_STRATEGY":
                v = AL_STRATEGY(int(v))
            elif k == "EXP_DATASET":
                v = DATASET(int(v))
            elif k == "EXP_LEARNER_MODEL":
                v = LEARNER_MODEL(int(v))

            if str(self.__annotations__[k]).endswith("class 'int'>"):
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
                str(v) == "typing.List[<class 'int'>]"
                or str(v) == "typing.Union[typing.List[int"
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
            elif (
                str(v) == "typing.List[resources.data_types.COMPUTED_METRIC]"
                or str(v) == "typing.List[resources.data_types.SAMPLES_CATEGORIZER]"
            ):
                full_str = str(v).split("[")[1][:-1].split(".")
                module_str = ".".join(full_str[:-1])
                class_str = full_str[-1]
                v_class = getattr(sys.modules[module_str], class_str)

                # allow all possible integer values from the enum classes
                choices = [e.name for e in v_class]  # type: ignore
                choices.append("_ALL")
                nargs = "*"
                arg_type = str
            elif str(v) == "typing.Union[str, pathlib.Path]":
                arg_type = str
            elif k == "EVA_METRICS_TO_CORRELATE":
                nargs = "*"
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
