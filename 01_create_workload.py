from calendar import c
import itertools
from pathlib import Path
import pandas as pd
from misc.config import Config
from misc.logging import log_it
import os


def create_workload(config: Config) -> None:
    # check results
    if os.path.isfile(config.RESULTS_FILE_PATH):
        result_df = pd.read_csv(
            config.RESULTS_FILE_PATH,
            index_col=None,
            usecols=["dataset_id", "strategy_id", "dataset_random_seed"],
        )
    else:
        result_df = pd.DataFrame(
            data=None, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
        )

    missing_ids = []

    for dataset_id, strategy_id, dataset_random_seed in itertools.product(
        config.EXP_DATASETS,
        config.EXP_STRATEGIES,
        config.EXP_RANDOM_SEEDS,
        repeat=1,
    ):
        if (
            len(
                result_df.loc[
                    (result_df["dataset_id"] == dataset_id)
                    & (result_df["strategy_id"] == strategy_id)
                    & (result_df["dataset_random_seed"] == dataset_random_seed)
                ]
            )
            == 0
        ):
            missing_ids.append([dataset_id, strategy_id, dataset_random_seed])

    random_seed_df = pd.DataFrame(
        data=missing_ids, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
    )

    random_seed_df.to_csv(config.WORKLOAD_FILE_PATH, header=True)
    config.save_to_file()


def _write_template_file(
    config: Config, template_name: Path, destination_path: Path
) -> None:
    pass


def create_AL_experiment_slurm_files(config: Config) -> None:
    print(config.OUTPUT_PATH)
    _write_template_file(
        config, Path("slurm_templates/parallel.sh"), config.EXPERIMENT_SLURM_FILE_PATH
    )
    # read in template

    # put in config parameters

    # write out slurm file
    pass


def create_AL_experiment_bash_files(config: Config) -> None:

    pass


def create_run_files(config: Config) -> None:
    # create rsync/slurm start file
    # create local run file
    pass


# usage example: python 01_create_workload.py --EXP_DATASETS 1,2,3,4,5,6 --EXP_STRATEGIES 5,10 --EXP_RANDOM_SEEDS 100

config = Config()

create_workload(config)
create_AL_experiment_slurm_files(config)
create_AL_experiment_bash_files(config)
create_run_files(config)
