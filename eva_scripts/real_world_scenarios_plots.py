import ast
import multiprocessing
import subprocess
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from resources.data_types import LEARNER_MODEL

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)
targets_to_evaluate = [
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_START_POINT",
]

for target_to_evaluate in targets_to_evaluate:
    results = pd.read_csv(
        Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/real_single_scenarios_{target_to_evaluate}_correlated.csv"
        )
    )

    jobs = list(results.iterrows())
    print(len(jobs))

    def do_stuff_wrapper(result):
        try:
            col_ix = [str(ccc) for ccc in ast.literal_eval(result["keys"])]
            data = ast.literal_eval(result["data"])
        except ValueError:
            print("nan")
            data = ast.literal_eval(result["data"].replace("nan", "0"))
            # data = [[ccc if ccc != 1000 else np.nan for ccc in ddd] for ddd in data]
        single_results_df = pd.DataFrame(columns=col_ix, data=data, index=col_ix)

        single_results_df = single_results_df.sort_index(axis=0)
        single_results_df = single_results_df.sort_index(axis=1)
        return single_results_df.to_numpy()

    dfs = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(do_stuff_wrapper)(result) for _, result in jobs
    )
    res = None
    for df in dfs:
        if res is None:
            res = df.copy()
        else:
            res += df
    res /= len(dfs)
    res *= 100
    print(res)

    keys = natsorted([str(ccc) for ccc in ast.literal_eval(results.iloc[0]["keys"])])

    if target_to_evaluate == "EXP_LEARNER_MODEL":
        keys = [
            LEARNER_MODEL(int(kkk)).name if kkk != "gold" else "gold" for kkk in keys
        ]
    elif target_to_evaluate == "EXP_DATASET":
        keys = [DATASET(int(kkk)).name if kkk != "gold" else "gold" for kkk in keys]

    save_correlation_plot(
        data=res,
        title=f"real/{target_to_evaluate}",
        keys=keys,
        config=config,
    )
