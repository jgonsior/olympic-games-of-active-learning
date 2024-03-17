import multiprocessing
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from misc.helpers import (
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

standard_metric = "weighted_f1-score"

targets_to_evaluate = [
    "EXP_BATCH_SIZE",
    "EXP_LEARNER_MODEL",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "EXP_STRATEGY",
    "EXP_START_POINT",
]

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

ts_orig = ts.copy()

for target_to_evaluate in targets_to_evaluate:
    correlation_data_path = Path(
        config.OUTPUT_PATH / f"plots/{target_to_evaluate}.parquet"
    )
    log_and_time(target_to_evaluate)
    if correlation_data_path.exists():
        corrmat = pd.read_parquet(correlation_data_path)
        print(corrmat)
        keys = corrmat.columns
        corrmat = corrmat.to_numpy()
        print("hui")
    else:
        ts = ts_orig.copy()

        fingerprint_cols = list(ts.columns)
        fingerprint_cols.remove("metric_value")
        fingerprint_cols.remove(target_to_evaluate)
        print(ts.dtypes)
        ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
            lambda row: "_".join([str(rrr) for rrr in row]), axis=1
        )

        log_and_time("Done fingerprinting")

        for fg_col in fingerprint_cols:
            del ts[fg_col]

        # ts = ts.sort_values(by="fingerprint")
        print(ts)
        # log_and_time("Done sorting")

        shared_fingerprints = None
        for target_value in ts[target_to_evaluate].unique():
            tmp_fingerprints = set(
                ts.loc[ts[target_to_evaluate] == target_value]["fingerprint"].to_list()
            )

            if shared_fingerprints is None:
                shared_fingerprints = tmp_fingerprints
            else:
                shared_fingerprints = shared_fingerprints.intersection(tmp_fingerprints)

        log_and_time(
            f"Done calculating shared fingerprints - {len(shared_fingerprints)}"
        )

        limited_ts = {}
        for target_value in ts[target_to_evaluate].unique():
            limited_ts[target_value] = ts.loc[
                (ts["fingerprint"].isin(shared_fingerprints))
                & (ts[target_to_evaluate] == target_value)
            ]["metric_value"].to_numpy()

        log_and_time("Done indexing ts")

        limited_ts_np = np.array([*limited_ts.values()])

        corrmat = np.corrcoef(limited_ts_np)
        log_and_time("Done correlation computations")

        keys = [*limited_ts.keys()]

    save_correlation_plot(
        data=corrmat, title=target_to_evaluate, keys=keys, config=config
    )
