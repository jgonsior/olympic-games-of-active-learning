import ast
import multiprocessing
import subprocess
import sys
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path

from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)

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

    jobs = list(results.iterrows())[:3]

    print(len(jobs))

    def do_stuff_wrapper(result):
        try:
            col_ix = [str(ccc) for ccc in ast.literal_eval(result["keys"])]
            data = ast.literal_eval(result["data"])
        except ValueError:
            data = ast.literal_eval(result["data"].replace("nan", "1000"))
            data = [[ccc if ccc != 1000 else np.nan for ccc in ddd] for ddd in data]
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

    print(res)

    save_correlation_plot(
        data=corrmat,
        title=f"single_hyperparameter/{target_to_evaluate}/{standard_metric}",
        keys=keys,
        config=config,
    )
