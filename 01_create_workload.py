import sys

sys.dont_write_bytecode = True
import os
import stat
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from jinja2 import Template
from sklearn.model_selection import ParameterGrid
from datasets import DATASET, load_dataset

from misc.config import Config
from misc.logging import log_it
from resources.data_types import (
    al_strategies_which_only_support_binary_classification,
    LEARNER_MODEL,
    al_strategies_which_require_decision_boundary_model,
    learner_models_to_classes_mapping,
    al_strategies_not_suitable_for_hpc,
)

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)


# determine config parameters which are to be used -> they all start with EXP_ and have a typing hint of [List[XXX]]
def _determine_exp_grid_parameters(config: Config) -> List[str]:
    result_list: List[str] = []

    for k, v in Config.__annotations__.items():
        if k.startswith("EXP_GRID_") and str(v).startswith("typing.List["):
            result_list.append(k)
    return result_list


def _generate_exp_param_grid(
    config: Config, exp_grid_params_names, INDEX_OFFSET=0
) -> pd.DataFrame:
    exp_param_grid = {
        exp_parameter: config.__getattribute__(exp_parameter)
        for exp_parameter in exp_grid_params_names
    }

    all_workloads = ParameterGrid(exp_param_grid)

    open_workload_df = pd.DataFrame(
        data=all_workloads,  # type: ignore
        columns=exp_grid_params_names,
    )

    open_workload_df.rename(
        columns=lambda s: s.replace("EXP_GRID_", "EXP_"), inplace=True  # type: ignore
    )

    if config.INCLUDE_RESULTS_FROM is not None:
        others_done_workload_df = None

        for other_exp_results_name in config.INCLUDE_RESULTS_FROM:
            # load done_workload_df from other results
            other_done_workload = pd.read_csv(Path(other_exp_results_name))
            if others_done_workload_df is None:
                others_done_workload_df = other_done_workload
            else:
                others_done_workload_df = pd.concat(
                    [others_done_workload_df, other_done_workload],
                    ignore_index=True,
                )
        length_before_removal_of_already_run_experiments = len(open_workload_df)
        hyperparameters = [
            h for h in open_workload_df.columns.to_list() if h not in ["EXP_UNIQUE_ID"]
        ]

        open_workload_df["ORIGINAL_INDEX"] = open_workload_df.index

        # merge dataframes
        merged_df = open_workload_df.merge(
            others_done_workload_df, how="inner", on=hyperparameters
        )
        merged_df.drop_duplicates(inplace=True)

        open_workload_df = open_workload_df.loc[
            ~open_workload_df.index.isin(merged_df["ORIGINAL_INDEX"])
        ]
        del open_workload_df["ORIGINAL_INDEX"]
        print(
            f"Reduced from {length_before_removal_of_already_run_experiments} to {len(open_workload_df)}"
        )

    # shuffle workload to ensure that really long jobs are not all running on the same node
    open_workload_df = open_workload_df.sample(frac=1).reset_index(drop=True)

    open_workload_df["EXP_UNIQUE_ID"] = open_workload_df.index + INDEX_OFFSET

    return open_workload_df


def create_workload(config: Config) -> List[int]:
    exp_grid_params_names = _determine_exp_grid_parameters(config)
    if os.path.isfile(config.OVERALL_DONE_WORKLOAD_PATH):
        # experiment has already been run, check which workloads are still missing
        done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

        open_workload_df = pd.read_csv(config.WORKLOAD_FILE_PATH)

        failed_workload_df = pd.read_csv(config.OVERALL_FAILED_WORKLOAD_PATH)

        # if new results exist -> recalculate hyperparameter grid
        # remove existing workloads
        # new ones get new experiment_ids
        # done!
        if config.RECALCULATE_UPDATED_EXP_GRID:
            print("Using updated config")
            new_open_workload_df = _generate_exp_param_grid(
                config,
                exp_grid_params_names,
                max(
                    done_workload_df["EXP_UNIQUE_ID"].max(),
                    open_workload_df["EXP_UNIQUE_ID"].max(),
                    failed_workload_df["EXP_UNIQUE_ID"].max(),
                )
                + 1,
            )

            new_open_workload_df.rename(
                columns={"EXP_UNIQUE_ID": "ORIGINAL_INDEX_NEW"}, inplace=True
            )

            hyperparameters = [
                h
                for h in open_workload_df.columns.to_list()
                if h not in ["EXP_UNIQUE_ID"]
            ]
            open_workload_df.rename(
                columns={"EXP_UNIQUE_ID": "ORIGINAL_INDEX_OLD"}, inplace=True
            )

            merged_df = open_workload_df.merge(
                new_open_workload_df,
                how="outer",
                on=hyperparameters,
            )

            merged_df.drop_duplicates(
                inplace=True,
                subset=merged_df.columns.difference(
                    ["ORIGINAL_INDEX_NEW", "ORIGINAL_INDEX_OLD"]
                ),
            )

            merged_df["MERGED_INDEX"] = merged_df["ORIGINAL_INDEX_OLD"].fillna(
                merged_df["ORIGINAL_INDEX_NEW"]
            )
            del merged_df["ORIGINAL_INDEX_NEW"]
            del merged_df["ORIGINAL_INDEX_OLD"]
            merged_df.rename(columns={"MERGED_INDEX": "EXP_UNIQUE_ID"}, inplace=True)
            open_workload_df = merged_df

            print(open_workload_df["EXP_UNIQUE_ID"])

            open_workload_df["EXP_UNIQUE_ID"] = open_workload_df[
                "EXP_UNIQUE_ID"
            ].astype(int)

        def _remove_right_from_left_workload(
            left: pd.DataFrame, right: pd.DataFrame
        ) -> pd.DataFrame:
            if "error" in right.columns:
                del right["error"]

            merge = left.merge(
                right,
                on=[c for c in left.columns.difference(["EXP_UNIQUE_ID"])],
                how="outer",
                indicator="source",
            )
            result = merge[merge.source.eq("left_only")].drop("source", axis=1)
            del result["EXP_UNIQUE_ID_y"]
            result.rename(columns={"EXP_UNIQUE_ID_x": "EXP_UNIQUE_ID"}, inplace=True)
            return result

        print(f"{len(open_workload_df)} - before removal")
        if not config.RERUN_FAILED_WORKLOADS:
            open_workload_df = _remove_right_from_left_workload(
                open_workload_df, failed_workload_df
            )
            print(f"{len(open_workload_df)} - removed failed")
        else:
            # only rerun those having MLP failure
            failed_workload_df = failed_workload_df[
                ~failed_workload_df["error"].isin(
                    [
                        # "<class 'sklearn.exceptions.ConvergenceWarning'>",
                        "<class 'OSError'>",
                        "<class 'BrokenPipeError'> ",
                    ]
                )
            ]
            open_workload_df = _remove_right_from_left_workload(
                open_workload_df, failed_workload_df
            )
            print(f"{len(open_workload_df)} - removed only some failed")

        # remove already run workloads
        open_workload_df = _remove_right_from_left_workload(
            open_workload_df, done_workload_df
        )
        print(f"{len(open_workload_df)} - removed already run")

        # remove workloads resulting in oom
        oom_workload_df = pd.read_csv(config.OVERALL_STARTED_OOM_WORKLOAD_PATH)
        open_workload_df = _remove_right_from_left_workload(
            open_workload_df, oom_workload_df
        )
        print(f"{len(open_workload_df)} - removed oom")
    else:
        open_workload_df = _generate_exp_param_grid(config, exp_grid_params_names)

    print(len(open_workload_df))
    print("Removing strategies which only work in binary case")
    # remove all workloads which do not work
    datasets_which_are_binary_only = []
    for dataset_id in open_workload_df["EXP_DATASET"].unique():
        dataset_df, _ = load_dataset(DATASET(dataset_id), config)

        if len(dataset_df["LABEL_TARGET"].unique()) > 2:
            datasets_which_are_binary_only.append(dataset_id)

    for binary_only_strategy in al_strategies_which_only_support_binary_classification:
        for binary_dataset_id in datasets_which_are_binary_only:
            open_workload_df = open_workload_df.drop(
                open_workload_df[
                    (open_workload_df["EXP_DATASET"] == binary_dataset_id)
                    & (open_workload_df["EXP_STRATEGY"] == binary_only_strategy)
                ].index
            )

    print(len(open_workload_df))
    print(
        "Remove strategies and ml_model combinations which only work with decision boundary"
    )
    ml_models_which_do_nothave_decision_boundary_function: List[LEARNER_MODEL] = []
    for ml_model_id in open_workload_df["EXP_LEARNER_MODEL"].unique():
        ml_model_class = learner_models_to_classes_mapping[LEARNER_MODEL(ml_model_id)]
        if not hasattr(ml_model_class, "decision_function"):
            ml_models_which_do_nothave_decision_boundary_function.append(ml_model_id)

    for (
        decision_boundary_requiring_al_class
    ) in al_strategies_which_require_decision_boundary_model:
        for (
            ml_model_withoud_decision_boundary_function
        ) in ml_models_which_do_nothave_decision_boundary_function:
            open_workload_df = open_workload_df.drop(
                open_workload_df[
                    (
                        open_workload_df["EXP_LEARNER_MODEL"]
                        == ml_model_withoud_decision_boundary_function
                    )
                    & (
                        open_workload_df["EXP_STRATEGY"]
                        == decision_boundary_requiring_al_class
                    )
                ].index
            )

    print(len(open_workload_df))
    print("Removing workloads using LBFGS_MLP")
    open_workload_df = open_workload_df[
        open_workload_df["EXP_LEARNER_MODEL"] != LEARNER_MODEL.LBFGS_MLP
    ]

    print(len(open_workload_df))
    if config.SEPARATE_HPC_LOCAL_WORKLOAD:
        non_hpc_workload_df = open_workload_df[
            open_workload_df["EXP_STRATEGY"].isin(al_strategies_not_suitable_for_hpc)
        ]

        # remove non_hpc_workloads
        open_workload_df = open_workload_df[
            ~open_workload_df["EXP_UNIQUE_ID"].isin(
                non_hpc_workload_df["EXP_UNIQUE_ID"]
            )
        ]
        non_hpc_workload_df.to_csv(config.NON_HPC_WORKLOAD_FILE_PATH, index=None)

    open_workload_df.to_csv(config.WORKLOAD_FILE_PATH, index=None)
    config.save_to_file()
    log_it(f"Created workload of {len(open_workload_df)}")
    return open_workload_df.EXP_UNIQUE_ID.to_numpy()


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
        Path("resources/slurm_templates/slurm_parallel.sh"),
        config.EXPERIMENT_SLURM_FILE_PATH,
        array=True,
        PYTHON_FILE="/02_run_experiment.py",
        START=0,
        END=int(workload_amount / config.SLURM_ITERATIONS_PER_BATCH),
        CLI_ARGS="",
        APPEND_OUTPUT_PATH=False,
        timeout_duration=config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT
        * (config.EXP_GRID_NUM_QUERIES[0] + 1),
    )

    _write_template_file(
        config,
        Path("resources/slurm_templates/chain_job.sh"),
        config.EXPERIMENT_SLURM_CHAIN_JOB,
    )
    _chmod_u_plus_x(config.EXPERIMENT_SLURM_CHAIN_JOB)

    _write_template_file(
        config,
        Path("resources/slurm_templates/tar_slurm.sh"),
        config.EXPERIMENT_SLURM_TAR_PATH,
        array=False,
    )


def create_AL_experiment_bash_files(config: Config, unique_ids: List[int]) -> None:
    _write_template_file(
        config,
        Path("resources/slurm_templates/02b_run_bash_parallel.py.jinja"),
        config.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH,
        PYTHON_FILE="02_run_experiment.py",
        START=0,
        END=len(unique_ids),
        timeout_duration=config.EXP_QUERY_SELECTION_RUNTIME_SECONDS_LIMIT
        * (config.EXP_GRID_NUM_QUERIES[0] + 1),
    )
    _chmod_u_plus_x(config.EXPERIMENT_PYTHON_PARALLEL_BASH_FILE_PATH)


def create_run_files(config: Config) -> None:
    _write_template_file(
        config,
        Path("resources/slurm_templates/sync_and_run.sh"),
        config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH,
    )
    _chmod_u_plus_x(config.EXPERIMENT_SYNC_AND_RUN_FILE_PATH)


config = Config()

unique_ids = create_workload(config)
create_AL_experiment_slurm_files(config, len(unique_ids))
create_AL_experiment_bash_files(config, unique_ids)
create_run_files(config)

_write_template_file(
    config,
    Path("resources/slurm_templates/update_pipenv_dep.slurm"),
    config.EXPERIMENT_UPDATE_SLURM_DEP_PATH,
)
_write_template_file(
    config,
    Path("resources/slurm_templates/install_pipenv_dep.slurm"),
    config.EXPERIMENT_INSTALL_SLURM_DEP_PATH,
)
_write_template_file(
    config,
    Path("resources/slurm_templates/02c_gzip_results.sh.slurm"),
    config.EXPERIMENT_SLURM_GZIP_RESULTS_PATH,
)

_chmod_u_plus_x(config.EXPERIMENT_SLURM_GZIP_RESULTS_PATH)
