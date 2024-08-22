import multiprocessing
import subprocess
import sys
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

standard_metric = "weighted_f1-score"

standard_metrics = [
    f"{fff}{standard_metric}"
    for fff in [
        "full_auc_",
        "first_5_",
        "final_value_",
        "last_5_",
        # "learning_stability_5_",
        # "learning_stability_10_",
        "ramp_up_auc_",
        "plateau_auc_",
    ]
]

standard_metrics.append(standard_metric)
# standard_metrics = [standard_metric]

for standard_metric in standard_metrics:
    log_and_time(f"Starting {standard_metric}")

    targets_to_evaluate = [
        "EXP_STRATEGY",
        "EXP_LEARNER_MODEL",
        "EXP_BATCH_SIZE",
        "EXP_DATASET",
        "EXP_TRAIN_TEST_BUCKET_SIZE",
        "EXP_START_POINT",
    ]

    if not Path(config.CORRELATION_TS_PATH / f"{standard_metric}.parquet").exists():
        unsorted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.unsorted.csv"
        unparqueted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.to_parquet.csv"

        if not unsorted_f.exists() and not unparqueted_f.exists():
            log_and_time("Create selected indices ts")
            create_fingerprint_joined_timeseries_csv_files(
                metric_names=[standard_metric], config=config
            )

        if not unparqueted_f.exists():
            log_and_time("Created, now sorting")
            command = f"sort -T {config.CORRELATION_TS_PATH} --parallel {multiprocessing.cpu_count()} {unsorted_f} -o {config.CORRELATION_TS_PATH}/{standard_metric}.to_parquet.csv"
            print(command)
            subprocess.run(command, shell=True, text=True)
            unsorted_f.unlink()

        log_and_time("sorted, now parqueting")
        ts = pd.read_csv(
            unparqueted_f,
            header=None,
            index_col=False,
            delimiter=",",
            names=[
                "EXP_DATASET",
                "EXP_STRATEGY",
                "EXP_START_POINT",
                "EXP_BATCH_SIZE",
                "EXP_LEARNER_MODEL",
                "EXP_TRAIN_TEST_BUCKET_SIZE",
                "ix",
                "EXP_UNIQUE_ID_ix",
                "metric_value",
            ],
        )
        ts["metric_value"] = ts["metric_value"].apply(
            lambda xxx: (
                np.fromstring(
                    str(xxx).removeprefix("[").removesuffix("]"),
                    dtype=np.int32,
                    sep=",",
                )
            )
        )

        f = Path(config.CORRELATION_TS_PATH / f"{standard_metric}.parquet")
        ts.to_parquet(f)
        unparqueted_f.unlink()

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
            config.OUTPUT_PATH
            / f"plots/single_hyperparameter/{target_to_evaluate}/{standard_metric}.parquet"
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

            if type(ts.iloc[0]["metric_value"]) == np.ndarray:
                ts["metric_value"] = ts["metric_value"].parallel_apply(
                    lambda aaa: aaa[0]
                )

            # log_and_time("Done sorting")
            shared_fingerprints = None
            for target_value in ts[target_to_evaluate].unique():
                tmp_fingerprints = set(
                    ts.loc[ts[target_to_evaluate] == target_value][
                        "fingerprint"
                    ].to_list()
                )

                if shared_fingerprints is None:
                    shared_fingerprints = tmp_fingerprints
                else:
                    shared_fingerprints = shared_fingerprints.intersection(
                        tmp_fingerprints
                    )

            log_and_time(
                f"Done calculating shared fingerprints - {len(shared_fingerprints)}"
            )

            limited_ts = {}
            for target_value in ts[target_to_evaluate].unique():
                limited_ts[target_value] = ts.loc[
                    (ts[target_to_evaluate] == target_value)
                    & (ts["fingerprint"].isin(shared_fingerprints))
                ]["metric_value"].to_numpy()

            log_and_time("Done indexing ts")

            limited_ts_np = np.array([*limited_ts.values()])

            corrmat = np.corrcoef(limited_ts_np)
            log_and_time("Done correlation computations")

            keys = [*limited_ts.keys()]

            save_correlation_plot(
                data=corrmat,
                title=f"single_hyperparameter/{target_to_evaluate}/single_hyper_{target_to_evaluate}_{standard_metric}",
                keys=keys,
                config=config,
            )
