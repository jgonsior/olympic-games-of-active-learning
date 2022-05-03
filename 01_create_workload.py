import itertools
import multiprocessing
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import pandas as pd
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed

# determine config parameters which are to be used -> they all start with EXP_ and have a typing hint of [List[XXX]]
def _determine_exp_grid_parameters(config: Config) -> List[str]:
    result_list: List[str] = []

    for k, v in Config.__annotations__.items():
        if k.startswith("EXP_GRID_") and str(v).startswith("typing.List["):
            result_list.append(k)
    return result_list


def create_workload(config: Config) -> None:
    exp_grid_params_names = _determine_exp_grid_parameters(config)

    if os.path.isfile(config.DONE_WORKLOAD_PATH):
        result_df = pd.read_csv(
            config.DONE_WORKLOAD_PATH,
        )
    else:
        result_df = pd.DataFrame(data=None, columns=exp_grid_params_names)

    missing_workloads = []

    exp_param_grid = {
        exp_parameter: config.__getattribute__(exp_parameter)
        for exp_parameter in exp_grid_params_names
    }

    def _check_if_workload_has_already_been_run(single_workload: Dict[str, Any]):
        selection_key = ()
        for k, v in single_workload.items():
            selection_key &= result_df[k] == v
        possible_existing_results_row = result_df.loc[selection_key]

        if len(possible_existing_results_row) == 0:
            return single_workload

    missing_workloads = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(_check_if_workload_has_already_been_run)(single_workload)
        for single_workload in ParameterGrid(exp_param_grid)
    )

    random_seed_df = pd.DataFrame(
        data=missing_workloads,
        columns=exp_grid_params_names,
    )

    random_seed_df.rename(
        columns=lambda s: s.replace("EXP_GRID_", "EXP_"), inplace=True  # type: ignore
    )

    random_seed_df.to_csv(config.WORKLOAD_FILE_PATH, index=None)
    config.save_to_file()
    log_it(f"Created workload of {len(random_seed_df)}")


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


# TODO wrapper schreiben, der mehrere random seeds aus der workload datei von einem worker ausführt
def create_AL_experiment_slurm_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/slurm_parallel.sh"),
        config.EXPERIMENT_SLURM_FILE_PATH,
        array=True,
        PYTHON_FILE="02_run_experiment.py",
        START=config.EXP_GRID_RANDOM_SEED[0],
        END=int(config.EXP_GRID_RANDOM_SEED[-1] / config.SLURM_ITERATIONS_PER_BATCH),
        CLI_ARGS="",
        APPEND_OUTPUT_PATH=False,
    )


def create_AL_experiment_bash_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/bash_parallel_runner.sh"),
        config.EXPERIMENT_BASH_FILE_PATH,
        PYTHON_FILE="02_run_experiment.py",
        START=config.EXP_GRID_RANDOM_SEED[0],
        END=int(config.EXP_GRID_RANDOM_SEED[-1] / config.SLURM_ITERATIONS_PER_BATCH),
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
