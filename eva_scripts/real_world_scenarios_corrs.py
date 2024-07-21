import ast
import copy
import multiprocessing
import random
from re import T
import subprocess
import sys

from misc.plotting import _rename_strategy

sys.dont_write_bytecode = True


from datasets import DATASET
from resources.data_types import AL_STRATEGY, LEARNER_MODEL
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import scipy
from datasets import DATASET
from misc.helpers import (
    append_and_create,
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
)
from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

prepare_eva_pathes("real_single_scenarios_corr", config)

if config.EVA_MODE == "create":

    if Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_decomposed.csv",
    ).exists():
        df = pd.read_csv(
            Path(
                config.OUTPUT_PATH
                / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_decomposed.csv",
            )
        )
    else:
        from pandarallel import pandarallel

        pandarallel.initialize(
            nb_workers=multiprocessing.cpu_count(),
            progress_bar=True,
            use_memory_fs=False,
        )

        df = pd.read_csv(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios.csv",
        )
        print(df)

        def _decompose_titles(row: pd.Series) -> pd.Series:
            name = ast.literal_eval(
                str(row["Unnamed: 0"]).removeprefix("real_single_scenarios: ")
            )
            row["EXP_ID"] = name[0]
            row["EXP_PARAM"] = name[1]
            row["EXP_PARAM_VALUE"] = name[2]
            row["EXP_AMOUNT_OF_OTHER_PARAMS"] = name[3]
            del row["Unnamed: 0"]
            return row

        df = df.parallel_apply(_decompose_titles, axis=1)
        df.to_csv(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_decomposed.csv",
            index=False,
        )

    jobs = list(enumerate(df["EXP_ID"].unique().tolist()))

    # one job -> one parameter with all its tpi values --> the first value of the tuple and the second to last one

    create_workload(
        jobs,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1000,
        SCRIPTS_PATH="eva_scripts",
        SLURM_NR_THREADS=1,
        script_type="eva_scripts",
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:
    df = pd.read_csv(
        Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_decomposed.csv",
        )
    )
    del df["EXP_AMOUNT_OF_OTHER_PARAMS"]

    # add gold standard
    gold_standard = pd.read_csv(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/standard_metric.csv",
        index_col=0,
    )
    gold_standard = (
        gold_standard.loc["standard_metric: full_auc_weighted_f1-score"].to_frame().T
    )
    gold_standard["EXP_PARAM_VALUE"] = "gold"

    def _run_single_metric(
        ix, hyperparameter_target_value, config: Config, additional_value=None
    ):
        df: pd.DataFrame = config._df

        df = df.loc[df["EXP_ID"] == hyperparameter_target_value]
        del df["EXP_ID"]

        if len(df["EXP_PARAM"].unique()) > 1:
            print("Oh oh" * 10000)
            exit(-1)
        hyperparameter_name = df["EXP_PARAM"].unique()[0]
        del df["EXP_PARAM"]
        df = pd.concat([df, gold_standard], ignore_index=True)
        df.set_index("EXP_PARAM_VALUE", inplace=True)

        corr_data = df.T.corr(method="kendall")
        result = corr_data.to_dict()
        result["ix"] = f"real_single_scenarios: {hyperparameter_target_value}"

        append_and_create(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_{hyperparameter_name}_correlated.csv",
            result,
        )

    config._df = df
    run_from_workload(do_stuff=_run_single_metric, config=config)
