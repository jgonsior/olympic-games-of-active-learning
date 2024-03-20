from itertools import combinations, combinations_with_replacement
import multiprocessing
import subprocess
import sys
import timeit
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

"""selected indices macht NUR Sinn bei learner model, strategy, start_point
bei batch_size werden unterschiedlich viele geholt, doof
bei dataset sind die indices ni vergleichbar
bei train_test_bucket sind die indices auch ni vergleichbar
aber bei start_point schon!
"""

targets_to_evaluate = [
    "EXP_STRATEGY",
    "EXP_LEARNER_MODEL",
    # "EXP_BATCH_SIZE",
    # "EXP_DATASET",
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
        # "EXP_UNIQUE_ID_ix",
        "metric_value",
    ],
)

ts_orig = ts.copy()

for target_to_evaluate in targets_to_evaluate:
    correlation_data_path = Path(
        config.OUTPUT_PATH / f"plots/{target_to_evaluate}_{standard_metric}.parquet"
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

        ts = ts.pivot(
            index="fingerprint", columns=target_to_evaluate, values="metric_value"
        )

        def _calculate_rank_correlations(r):
            js = []
            for c1, c2 in combinations(r.to_list(), 2):
                if np.isnan(c1).any() or np.isnan(c2).any():
                    js.append([0, 0, 0])
                else:
                    ken = kendalltau(c1, c2)
                    a = set(c1)
                    b = set(c2)
                    jaccard = len(a.intersection(b)) / len(a.union(b))

                    js.append([ken.statistic, ken.pvalue, jaccard])
            return pd.Series(js)

        jaccards = ts.parallel_apply(_calculate_rank_correlations, axis=1)
        jaccards.columns = [
            (ccc[0], ccc[1]) for ccc in combinations(ts.columns.to_list(), 2)
        ]
jaccard nur bis zum plateau punkt berechnen, ab da wird ja eh nur noch quatsch hinzugefügt -> neue metrik "selected indices plateau? oder selected_indices.parquet einlesen, und dann einmalig überall abcutten mittel parallel_apply?"
die single_hyperparameter_metric analyse über nach tmit f1_score, f1_score_full_auc, f1_score_rampu_up, f1_score_last_5 etc. --> darüber schauen welche metriken Sinn ergeben, für welche kann ich ähnliche strategien erkennen?
--> selected_indices für die ganzen auc werte erstellen!
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
                pd.DataFrame(data=corrmat).pivot(index=0, columns=1, values=2).fillna(1)
            ).to_numpy()

            keys = [ttt for ttt in ts.columns]

            save_correlation_plot(
                data=corrmat,
                title=f"{target_to_evaluate} - {standard_metric} - {rank_measure}",
                keys=keys,
                config=config,
            )
