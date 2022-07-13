from collections import Counter
import enum
import itertools
import multiprocessing
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import modin.pandas as pd
from datasets import DATASET
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed

from ressources.data_types import AL_STRATEGY, LEARNER_MODEL

# determine config parameters which are to be used -> they all start with EXP_ and have a typing hint of [List[XXX]]
def _determine_exp_grid_parameters(config: Config) -> List[str]:
    result_list: List[str] = []

    for k, v in Config.__annotations__.items():
        if k.startswith("EXP_GRID_") and str(v).startswith("typing.List["):
            result_list.append(k)
    return result_list


def _create_exp_grid(
    exp_strat_grid: List[Dict[AL_STRATEGY, Dict[str, List[Any]]]], config: Config
) -> List[str]:
    result: List[str] = []
    for a in exp_strat_grid:
        for b, c in a.items():
            kwargs = []
            for d, e in c.items():
                kwargs.append(
                    [f"{d}{config._EXP_STRATEGY_PARAM_VALUE_DELIM}{_x}" for _x in e]
                )
            for f in [
                config._EXP_STRATEGY_PARAM_PARAM_DELIM.join(_x)
                for _x in itertools.product(*kwargs)
            ]:
                result.append(f"{b}{config._EXP_STRATEGY_STRAT_PARAMS_DELIM}{f}")
    return result


def create_workload(config: Config) -> List[int]:
    exp_grid_params_names = _determine_exp_grid_parameters(config)

    if os.path.isfile(config.DONE_WORKLOAD_PATH):
        # experiment has already been run, check which worsloads are still missing
        done_workload_df = pd.read_csv(
            config.DONE_WORKLOAD_PATH,
        )

        open_workload_df = pd.read_csv(config.WORKLOAD_FILE_PATH)

        open_workload_df = open_workload_df.loc[
            ~open_workload_df.EXP_UNIQUE_ID.isin(done_workload_df.EXP_UNIQUE_ID)
        ]

    else:
        exp_param_grid = {
            exp_parameter: config.__getattribute__(exp_parameter)
            for exp_parameter in exp_grid_params_names
        }

        # convert EXP_GRID_STRATEGY with params into list of param objects
        exp_param_grid["EXP_GRID_STRATEGY"] = _create_exp_grid(
            exp_param_grid["EXP_GRID_STRATEGY"], config
        )
        all_workloads = ParameterGrid(exp_param_grid)

        open_workload_df = pd.DataFrame(
            data=all_workloads,  # type: ignore
            columns=exp_grid_params_names,
        )

        open_workload_df.rename(
            columns=lambda s: s.replace("EXP_GRID_", "EXP_"), inplace=True  # type: ignore
        )

        # shuffle workload to ensure that really long jobs are not all running on the same node
        open_workload_df = open_workload_df.sample(frac=1).reset_index(drop=True)

        open_workload_df["EXP_UNIQUE_ID"] = open_workload_df.index

    if config.INCLUDE_RESULTS_FROM is not None:
        others_done_workload_df = pd.DataFrame(columns=open_workload_df.columns)

        for other_exp_results_name in config.INCLUDE_RESULTS_FROM:
            # load done_workload_df from other results
            other_done_workload = pd.read_csv(Path(other_exp_results_name))
            others_done_workload_df = pd.concat(
                [others_done_workload_df, other_done_workload], ignore_index=True
            )

        hyperparameters = [
            h for h in open_workload_df.columns.to_list() if h not in ["EXP_UNIQUE_ID"]
        ]

        # convert strings to enums
        others_done_workload_df["EXP_DATASET"] = others_done_workload_df[
            "EXP_DATASET"
        ].apply(lambda x: DATASET[x.split(".")[1]])
        others_done_workload_df["EXP_LEARNER_MODEL"] = others_done_workload_df[
            "EXP_LEARNER_MODEL"
        ].apply(lambda x: LEARNER_MODEL[x.split(".")[1]])

        others_done_workload_df["EXP_STRATEGY_PARAMS"] = others_done_workload_df[
            "EXP_STRATEGY_PARAMS"
        ].apply(
            lambda x: x.replace("'", "")
            .replace(": ", config._EXP_STRATEGY_PARAM_VALUE_DELIM)
            .replace("{", "")
            .replace("}", "")
            .replace(", ", config._EXP_STRATEGY_PARAM_PARAM_DELIM)
        )

        # fancily encode exp_strategy
        others_done_workload_df["EXP_STRATEGY"] = others_done_workload_df[
            "EXP_STRATEGY"
        ].apply(lambda x: AL_STRATEGY[x.split(".")[1]])

        others_done_workload_df["EXP_STRATEGY"] = others_done_workload_df[
            "EXP_LEARNER_MODEL"
        ].apply(lambda x: f"{x}{config._EXP_STRATEGY_STRAT_PARAMS_DELIM}")

        others_done_workload_df["EXP_STRATEGY"] = (
            others_done_workload_df["EXP_STRATEGY"]
            + "_"
            + others_done_workload_df["EXP_STRATEGY_PARAMS"]
        )

        # convert wrong 1#_ back to 1
        others_done_workload_df["EXP_STRATEGY"] = others_done_workload_df[
            "EXP_STRATEGY"
        ].apply(lambda x: x.replace("#_", "#") if x.endswith("#_") else x)

        length_before_removal_of_already_run_experiments = len(open_workload_df)

        def _delete_already_run_row(row, open_workload_df):
            mask = True
            for hyper_parameter in hyperparameters:
                value = row[hyper_parameter]
                mask &= open_workload_df[hyper_parameter] == value

            open_workload_df = open_workload_df.loc[~mask]

        others_done_workload_df.apply(
            lambda x: _delete_already_run_row(x, open_workload_df)
        )

        """for _, row in others_done_workload_df.iterrows():
            mask = True
            for hyper_parameter in hyperparameters:
                value = row[hyper_parameter]
                mask &= open_workload_df[hyper_parameter] == value

            open_workload_df = open_workload_df.loc[~mask]
        """
        print(
            f"Reduced from {length_before_removal_of_already_run_experiments} to {len(open_workload_df)}"
        )

    open_workload_df.to_csv(config.WORKLOAD_FILE_PATH, index=None)
    config.save_to_file()
    log_it(f"Created workload of {len(open_workload_df)}")
    return open_workload_df.EXP_UNIQUE_ID.to_list()


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


def create_AL_experiment_slurm_files(config: Config, workload_amount: int) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/slurm_parallel.sh"),
        config.EXPERIMENT_SLURM_FILE_PATH,
        array=True,
        PYTHON_FILE="/02_run_experiment.py",
        START=0,
        END=int(workload_amount / config.SLURM_ITERATIONS_PER_BATCH),
        CLI_ARGS="",
        APPEND_OUTPUT_PATH=False,
    )

    _write_template_file(
        config,
        Path("slurm_templates/chain_job.sh"),
        config.EXPERIMENT_SLURM_CHAIN_JOB,
    )
    _chmod_u_plus_x(config.EXPERIMENT_SLURM_CHAIN_JOB)

    _write_template_file(
        config,
        Path("slurm_templates/tar_slurm.sh"),
        config.EXPERIMENT_SLURM_TAR_PATH,
        array=False,
    )


def create_AL_experiment_bash_files(config: Config, unique_ids: List[int]) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/02b_run_bash_parallel.py.jinja"),
        config.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH,
        PYTHON_FILE="02_run_experiment.py",
        START=0,
        END=len(unique_ids),
    )
    _chmod_u_plus_x(config.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH)


def create_run_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("slurm_templates/sync_and_run.sh"),
        config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH,
    )
    _chmod_u_plus_x(config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH)


config = Config()

unique_ids = create_workload(config)
create_AL_experiment_slurm_files(config, len(unique_ids))
create_AL_experiment_bash_files(config, unique_ids)
create_run_files(config)
