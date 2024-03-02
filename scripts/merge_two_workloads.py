import ast
import functools
import itertools
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from datasets import DATASET
from resources.data_types import AL_STRATEGY


sys.dont_write_bytecode = True

from misc.config import Config

from misc.helpers import (
    combine_results,
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
    get_df,
)

config = Config()


print(f"Merging {config.OUTPUT_PATH} and {config.SECOND_MERGE_PATH}")

prepare_eva_pathes("MERGE_TWO_WORKLOADS", config)

if config.EVA_MODE == "create":
    workload = [
        iii
        for iii in itertools.product(config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY)
    ]
    create_workload(
        workload,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1,
        SCRIPTS_PATH="scripts",
        SLURM_NR_THREADS=128,
        CLI_ARGS=f"--SECOND_MERGE_PATH {config.SECOND_MERGE_PATH}",
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:

    def do_stuff(exp_dataset, exp_strategy, config):
        csv_glob_list = sorted(
            [
                Path(ggg)
                for ggg in glob.glob(
                    config.SECOND_MERGE_PATH
                    + f"/{AL_STRATEGY(exp_strategy).name}/{DATASET(exp_dataset).name}/*.csv",
                    recursive=True,
                )
            ]
        )

        print(
            config.SECOND_MERGE_PATH
            + f"/{AL_STRATEGY(exp_strategy).name}/{DATASET(exp_dataset).name}/*.csv"
        )

        if len(csv_glob_list) == 0:
            print("empty")
            return

        from pandarallel import pandarallel

        pandarallel.initialize(progress_bar=True, nb_workers=20, use_memory_fs=True)

        for csv_file_name in csv_glob_list:
            print(csv_file_name)

            # if not "y_pred" in csv_file_name.name:
            #    continue

            csv_df = get_df(csv_file_name, config)

            if "y_pred" in csv_file_name.name:
                cols_with_indice_lists = csv_df.columns.difference(["EXP_UNIQUE_ID"])

                csv_df[cols_with_indice_lists] = (
                    csv_df[cols_with_indice_lists]
                    .fillna("[]")
                    .parallel_applymap(lambda x: ast.literal_eval(x))
                    #  .map(lambda x: ast.literal_eval(x))
                )

            original_csv_path = (
                config.OUTPUT_PATH
                / csv_file_name.parent.parent.name
                / csv_file_name.parent.name
                / csv_file_name.name
            )

            if "y_pred" in csv_file_name.name:
                xz_df = get_df(Path(str(original_csv_path) + ".xz.parquet"), config)
            else:
                xz_df = get_df(Path(str(original_csv_path)), config)

            # xz_df[cols_with_indice_lists] = (
            #    xz_df[cols_with_indice_lists]
            #    .fillna("[]")
            #    .parallel_applymap(lambda x: ast.literal_eval(x))
            #    #  .map(lambda x: ast.literal_eval(x))
            # )

            xz_df = pd.concat([xz_df, csv_df], ignore_index=True).drop_duplicates(
                subset="EXP_UNIQUE_ID"
            )

            if "Unnamed: " in xz_df.columns:
                del xz_df["Unnamed: 0"]

            if "y_pred" in csv_file_name.name:
                xz_df.to_parquet(Path(str(original_csv_path) + ".xz.parquet"))
            else:
                xz_df.to_csv(Path(str(original_csv_path) + ".xz"), index=False)

            csv_file_name.unlink()
        return None

    run_from_workload(do_stuff=do_stuff, config=config)
elif config.EVA_MODE == "combine":
    result_df = combine_results(config=config)

    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
    second_done_workload_df = pd.read_csv(
        config.SECOND_MERGE_PATH + "/" + config.OVERALL_DONE_WORKLOAD_PATH.name
    )

    merged_workload_df = pd.concat(
        [done_workload_df, second_done_workload_df], ignore_index=True
    ).drop_duplicates(subset="EXP_UNIQUE_ID")

    merged_workload_df.to_csv(config.OVERALL_DONE_WORKLOAD_PATH, index=False)
