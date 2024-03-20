import csv
import multiprocessing
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel


pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)
config = Config()
dataset_dependend_thresholds_df = pd.read_csv(
    config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH
)

print(dataset_dependend_thresholds_df)

selected_indices_ts = pd.read_parquet(
    config.CORRELATION_TS_PATH / f"selected_indices.parquet"
)

selected_indices_ts["EXP_UNIQUE_ID_ix"] = selected_indices_ts[
    "EXP_UNIQUE_ID_ix"
].parallel_apply(lambda x: int(x.removesuffix("_0")))

selected_indices_ts.rename(columns={"EXP_UNIQUE_ID_ix": "EXP_UNIQUE_ID"}, inplace=True)

selected_indices_ts = pd.merge(
    selected_indices_ts, dataset_dependend_thresholds_df, on="EXP_UNIQUE_ID", how="left"
)

selected_indices_ts.rename(
    columns={"EXP_UNIQUE_ID": "EXP_UNIQUE_ID_ix", "metric_value": "selected_indices"},
    inplace=True,
)

selected_indices_ts["EXP_UNIQUE_ID_ix"] = selected_indices_ts[
    "EXP_UNIQUE_ID_ix"
].parallel_apply(lambda x: f"{x}_0")

print(selected_indices_ts)

auc_names = [
    "ramp_up_auc_",
    "plateau_auc_",
    "first_5_",
    "last_5_",
]

for auc in auc_names:
    print(auc)
    match auc:
        case "first_5_":
            selected_indices_ts["metric_value"] = selected_indices_ts[
                "selected_indices"
            ].parallel_apply(lambda x: x[0:5])
        case "last_5_":
            selected_indices_ts["metric_value"] = selected_indices_ts[
                "selected_indices"
            ].parallel_apply(lambda x: x[-5:])
        case "ramp_up_auc_":
            selected_indices_ts["metric_value"] = selected_indices_ts[
                ["selected_indices", "cutoff_value"]
            ].parallel_apply(
                lambda x: x["selected_indices"][: x["cutoff_value"]], axis=1
            )
        case "plateau_auc_":
            selected_indices_ts["metric_value"] = selected_indices_ts[
                ["selected_indices", "cutoff_value"]
            ].parallel_apply(
                lambda x: x["selected_indices"][x["cutoff_value"] :], axis=1
            )

    cols_to_save = [
        ccc
        for ccc in selected_indices_ts.columns
        if ccc not in ["selected_indices", "cutoff_value"]
    ]

    selected_indices_ts[cols_to_save].to_parquet(
        config.CORRELATION_TS_PATH / f"{auc}selected_indices.parquet"
    )
