import itertools
from pathlib import Path
import stat
from typing import Any, Dict
from jinja2 import Template
import pandas as pd
from config.config import Config
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
    config: Config, template_path: Path, destination_path: Path, **kwargs
) -> None:
    template = Template(template_path.read_text())

    data: Dict[str, Any] = {**config.__dict__, **kwargs}

    rendered_template = template.render(**data)
    # log_it(rendered_template)
    destination_path.write_text(rendered_template)


def _chmod_u_plus_x(path: Path) -> None:
    st = os.stat(path)
    os.chmod(
        path,
        st.st_mode | stat.S_IEXEC,
    )


def create_AL_experiment_slurm_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/slurm_parallel.sh"),
        config.EXPERIMENT_SLURM_FILE_PATH,
        array=True,
        PYTHON_FILE="02_run_experiment.py",
        START=config.EXP_RANDOM_SEEDS[0],
        END=int(config.EXP_RANDOM_SEEDS[-1] / config.SLURM_ITERATIONS_PER_BATCH),
        CLI_ARGS="",
        APPEND_OUTPUT_PATH=False,
    )


def create_AL_experiment_bash_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/bash_parallel_runner.sh"),
        config.EXPERIMENT_BASH_FILE_PATH,
        PYTHON_FILE="02_run_experiment.py",
        START=config.EXP_RANDOM_SEEDS[0],
        END=int(config.EXP_RANDOM_SEEDS[-1] / config.SLURM_ITERATIONS_PER_BATCH),
    )
    _chmod_u_plus_x(config.EXPERIMENT_BASH_FILE_PATH)


def create_run_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/sync_and_run.sh"),
        config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH,
    )
    _chmod_u_plus_x(config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH)


config = Config()

create_workload(config)
create_AL_experiment_slurm_files(config)
create_AL_experiment_bash_files(config)
create_run_files(config)
