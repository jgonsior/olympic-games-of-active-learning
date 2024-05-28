from itertools import combinations, combinations_with_replacement
import multiprocessing
import subprocess
import sys
import timeit
from turtle import st
from typing import Literal
from git import Object
from scipy.stats import kendalltau
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import jaccard_score
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

standard_metric = "selected_indices"

for auc_prefix in [
    "final_value_",
    "ramp_up_auc_",
    "plateau_auc_",
    "full_auc_",
    "first_5_",
    "last_5_",
]:
    log_and_time(f"Calculating for {standard_metric}")
    targets_to_evaluate = [
        "EXP_STRATEGY",  # gibt es strategie ähnlichkeiten?
        "EXP_LEARNER_MODEL",  # gibt es ähnlichkeiten zwischen den learner modellen?
        # "EXP_BATCH_SIZE",
        # "EXP_DATASET",
        # "EXP_TRAIN_TEST_BUCKET_SIZE",
        "EXP_START_POINT",  # wurden vom selben startpunkt ausgehend diesselben datenpunkte ausgewählt?
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
        print(unparqueted_f)
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
        print(ts["metric_value"])
        ts["metric_value"] = ts["metric_value"].apply(
            lambda xxx: (
                np.fromstring(
                    xxx.removeprefix("[").removesuffix("]"),
                    dtype=np.int32,
                    sep=" ",
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
            # "ix",
            "EXP_UNIQUE_ID_ix",
            "metric_value",
        ],
    )

    ts_orig = ts.copy()

    for target_to_evaluate in targets_to_evaluate:
        correlation_data_path = Path(
            config.OUTPUT_PATH
            / f"plots/single_hyperparameter/{target_to_evaluate}/{auc_prefix}_{standard_metric}_statistic.parquet"
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

            ts["EXP_UNIQUE_ID"] = ts["EXP_UNIQUE_ID_ix"].parallel_apply(
                lambda e_ix: int(e_ix.split("_")[0])
            )
            del ts["EXP_UNIQUE_ID_ix"]

            fingerprint_cols = list(ts.columns)
            fingerprint_cols.remove("metric_value")
            fingerprint_cols.remove("EXP_UNIQUE_ID")
            fingerprint_cols.remove(target_to_evaluate)

            ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
                lambda row: "_".join([str(rrr) for rrr in row]), axis=1
            )

            log_and_time("Done fingerprinting")

            for fg_col in fingerprint_cols:
                del ts[fg_col]

            # ts = ts.sort_values(by="fingerprint")
            # print(ts)
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

            ts = ts.loc[(ts["fingerprint"].isin(shared_fingerprints))]

            dataset_dependent_ramp_plateau_threshold_df = pd.read_csv(
                config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH
            )
            ts = ts.merge(
                dataset_dependent_ramp_plateau_threshold_df,
                left_on="EXP_UNIQUE_ID",
                right_on="EXP_UNIQUE_ID",
            )
            del ts["EXP_UNIQUE_ID"]

            def _apply_ramp_up_or_plateau(
                ramp_up_or_plateau: Literal["ramp_up", "plateau"], row: pd.Series
            ) -> pd.Series:
                if ramp_up_or_plateau == "ramp_up":

                    range_start = row["cutoff_value"]
                    if np.isnan(range_start):
                        range_start = len(row) / 2

                    range_start = int(range_start)
                    row["metric_value"] = row["metric_value"][0:range_start].tolist()
                elif ramp_up_or_plateau == "plateau":
                    range_start = row["cutoff_value"]
                    if np.isnan(range_start):
                        range_start = len(row) / 2

                    range_start = int(range_start)
                    row["metric_value"] = row["metric_value"][range_start:]
                else:
                    print("ramp_up or plateau speeling error")
                    exit(-1)

                return row

            def _apply_ramp_up(row: pd.Series) -> pd.Series:
                return _apply_ramp_up_or_plateau(ramp_up_or_plateau="ramp_up", row=row)

            def _apply_plateau(row: pd.Series) -> pd.Series:
                return _apply_ramp_up_or_plateau(ramp_up_or_plateau="plateau", row=row)

            match auc_prefix:
                case "full_auc_":
                    ts["metric_value"] = ts["metric_value"].parallel_apply(
                        lambda lll: lll
                    )
                case "first_5_":
                    ts["metric_value"] = ts["metric_value"].parallel_apply(
                        lambda lll: lll[:5]
                    )
                case "last_5_":
                    ts["metric_value"] = ts["metric_value"].parallel_apply(
                        lambda lll: lll[-5:]
                    )
                case "final_value_":
                    ts["metric_value"] = ts["metric_value"].parallel_apply(
                        lambda lll: lll[-1:]
                    )
                case "ramp_up_auc_":
                    ts = ts.parallel_apply(_apply_ramp_up, axis=1)
                case "plateau_auc_":
                    ts = ts.parallel_apply(_apply_plateau, axis=1)
                case _:
                    ts = None
            del ts["cutoff_value"]

            ts = ts.pivot(
                index="fingerprint", columns=target_to_evaluate, values="metric_value"
            )

            def _calculate_rank_correlations(r):
                js = []
                for c1, c2 in combinations(r.to_list(), 2):
                    if np.isnan(c1).any() or np.isnan(c2).any():
                        js.append([0, 0, 0])
                    else:
                        a = set(c1)
                        b = set(c2)
                        jaccard = len(a.intersection(b)) / len(a.union(b))

                        if len(c1) != len(c2):
                            js.append([np.nan, np.nan, jaccard])
                        else:
                            ken = kendalltau(c1, c2)
                            js.append([ken.statistic, ken.pvalue, jaccard])
                return pd.Series(js)

            jaccards = ts.parallel_apply(_calculate_rank_correlations, axis=1)
            jaccards.columns = [
                (ccc[0], ccc[1]) for ccc in combinations(ts.columns.to_list(), 2)
            ]

            for rank_measure in ["statistic", "pvalue", "jaccard"]:
                if rank_measure == "statistic":
                    jaccards2 = jaccards.parallel_applymap(lambda x: x[0])
                elif rank_measure == "pvalue":
                    jaccards2 = jaccards.parallel_applymap(lambda x: x[1])
                elif rank_measure == "jaccard":
                    jaccards2 = jaccards.parallel_applymap(lambda x: x[2])

                sums = jaccards2.sum() / len(jaccards2)

                corrmat = []
                for ix, jaccards3 in sums.items():
                    c1 = ix[0]
                    c2 = ix[1]
                    corrmat.append((c1, c2, jaccards3))
                    corrmat.append((c2, c1, jaccards3))

                corrmat = (
                    pd.DataFrame(data=corrmat)
                    .pivot(index=0, columns=1, values=2)
                    .fillna(1)
                ).to_numpy()

                keys = [ttt for ttt in ts.columns]

                save_correlation_plot(
                    data=corrmat,
                    title=f"single_hyperparameter/{target_to_evaluate}/{auc_prefix}_{standard_metric}_{rank_measure}",
                    keys=keys,
                    config=config,
                )
