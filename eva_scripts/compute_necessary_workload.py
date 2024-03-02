from pathlib import Path
import sys
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import seaborn as sns

from misc.helpers import (
    combine_results,
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
    get_df,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

prepare_eva_pathes("MERGE_TWO_WORKLOADS", config)


standard_metric = "full_auc_macro_f1-score"

if config.EVA_MODE == "create":
    ts = pd.read_parquet(
        config.CORRELATION_TS_PATH / f"{standard_metric}.parquet",
        columns=[
            "EXP_DATASET",
            "EXP_STRATEGY",
            "EXP_START_POINT",
            "EXP_BATCH_SIZE",
            "EXP_LEARNER_MODEL",
            "EXP_TRAIN_TEST_BUCKET_SIZE",
            "ix",
            # "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )

    fingerprint_cols = list(ts.columns)
    fingerprint_cols.remove("metric_value")
    fingerprint_cols.remove("EXP_STRATEGY")

    ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
        lambda row: "_".join([str(rrr) for rrr in row]), axis=1
    )

    for fg_col in fingerprint_cols:
        del ts[fg_col]

    ts = ts.pivot(
        index="fingerprint", columns="EXP_STRATEGY", values="metric_value"
    ).reset_index()
    print(ts)
    # ts to parquet in workload folder??

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
    # load parquet folder line?
    def do_stuff(fingerprint, config):
        return None

    run_from_workload(do_stuff=do_stuff, config=config)
elif config.EVA_MODE == "combine":
    result_df = combine_results(config=config)

    cols_without_fingerprint = list(ts.columns)
    cols_without_fingerprint.remove("fingerprint")

    # np_data = ts[cols_without_fingerprint].to_numpy()

    # corrmatt = np.corrcoef(np_data)
    corrmatt = ts[cols_without_fingerprint].corr()
    print(corrmatt)

    save_correlation_plot(
        data=corrmatt,
        title="Necessary Workload",
        keys=ts["fingerprint"].to_list(),
        total=True,
        config=config,
    )
