import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)

sys.dont_write_bytecode = True

from misc.config import Config  # noqa: E402

config = Config()
from pandarallel import pandarallel  # noqa: E402

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

standard_metric = "weighted_f1-score"

standard_metrics = [
    f"{fff}{standard_metric}"
    for fff in [
        "full_auc_",
        # "first_5_",
        # "final_value_",
        # "last_5_",
        # "learning_stability_5_",
        # "learning_stability_10_",
        # "ramp_up_auc_",
        # gs"plateau_auc_",
    ]
]

# standard_metrics.append(standard_metric)
# standard_metrics = [standard_metric]

for standard_metric in standard_metrics:
    log_and_time(f"Starting {standard_metric}")

    targets_to_evaluate = [
        # "EXP_STRATEGY",
        "EXP_LEARNER_MODEL",
        # "EXP_BATCH_SIZE",
        # "EXP_DATASET",
        # "EXP_TRAIN_TEST_BUCKET_SIZE",
        # "EXP_START_POINT",
        # "EXP_START_POINT",
    ]

    if not Path(config.CORRELATION_TS_PATH / f"{standard_metric}.parquet").exists():
        unsorted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.unsorted.csv"
        unparqueted_f = config.CORRELATION_TS_PATH / f"{standard_metric}.to_parquet.csv"

        if not unsorted_f.exists() and not unparqueted_f.exists():
            print(standard_metrics)
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
        # for dataset_id in ts_orig["EXP_DATASET"].unique():

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
            # ts = ts.loc[ts["EXP_DATASET"] == dataset_id]

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

            def corrcoef_ci(X, confidence=0.95, rowvar=True):
                X = np.asarray(X)
                n = X.shape[0] if not rowvar else X.shape[1]
                R = np.corrcoef(X, rowvar=rowvar)

                # Fisher-z
                R_clip = np.clip(R, -0.999999999, 0.999999999)
                Z = np.arctanh(R_clip)
                se = 1.0 / np.sqrt(
                    max(n - 3, 1)
                )  # n<=3 -> CI wird [-1,1] (siehe SciPy-Doku)
                zcrit = norm.ppf(0.5 + confidence / 2)

                lo = np.tanh(Z - zcrit * se)
                hi = np.tanh(Z + zcrit * se)

                np.fill_diagonal(lo, 1.0)
                np.fill_diagonal(hi, 1.0)
                return R, lo, hi

            def corrcoef_with_pm(X, confidence=0.95, rowvar=True):
                X = np.asarray(X)
                n = X.shape[1] if rowvar else X.shape[0]
                if n < 4:
                    raise ValueError("Need at least n>=4 observations for Fisher-z CI.")

                R = np.corrcoef(X, rowvar=rowvar)
                R_clip = np.clip(R, -0.999999999, 0.999999999)
                Z = np.arctanh(R_clip)
                se = 1.0 / np.sqrt(n - 3)

                alpha = 1.0 - confidence
                if abs(confidence - 0.95) < 1e-12:
                    zcrit = 1.959963984540054
                else:
                    p = 1.0 - alpha / 2.0
                    zcrit = np.sqrt(2.0) * np.erfinv(2.0 * p - 1.0)

                lo = np.tanh(Z - zcrit * se)
                hi = np.tanh(Z + zcrit * se)

                pm = (hi - lo) / 2.0
                np.fill_diagonal(pm, 0.0)  # r with itself has no uncertainty shown
                return R, pm

            def corrcoef_with_se(X, rowvar=True):
                X = np.asarray(X)
                n = X.shape[1] if rowvar else X.shape[0]
                if n < 4:
                    raise ValueError("Need at least n>=4 observations.")

                R = np.corrcoef(X, rowvar=rowvar)
                se_r = (1.0 - R**2) / np.sqrt(n - 3)  # approx.
                np.fill_diagonal(se_r, 0.0)
                return R, se_r

            import matplotlib.pyplot as plt

            def plot_corr_with_ci(
                R, lo, hi, labels=None, title="Correlation with 95% CI"
            ):
                R = np.asarray(R)
                lo = np.asarray(lo)
                hi = np.asarray(hi)
                assert R.shape == lo.shape == hi.shape
                k = R.shape[0]

                if labels is None:
                    labels = [f"V{i + 1}" for i in range(k)]

                fig, ax = plt.subplots(figsize=(5.2, 4.6))
                im = ax.imshow(
                    R, vmin=-1, vmax=1
                )  # don't set colors explicitly per instructions

                ax.set_xticks(np.arange(k))
                ax.set_yticks(np.arange(k))
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)

                # Annotate each cell
                for i in range(k):
                    for j in range(k):
                        if i == j:
                            txt = "1.00"
                        else:
                            txt = f"{R[i, j]:.2f}\n[{lo[i, j]:.2f}, {hi[i, j]:.2f}]"
                        ax.text(j, i, txt, ha="center", va="center", fontsize=9)

                ax.set_title(title)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                return fig, ax

            corrmat = np.corrcoef(limited_ts_np)
            corrcoef_cis = corrcoef_ci(limited_ts_np)

            log_and_time("Done correlation computations")

            keys = [*limited_ts.keys()]
            print(corrcoef_cis)

            annot_lo_hi = np.zeros(corrcoef_cis[0].shape)

            print(annot_lo_hi)
            # exit(-1)

            save_correlation_plot(
                data=corrcoef_cis[0],
                title=f"single_hyperparameter/{target_to_evaluate}/single_hyper_{target_to_evaluate}_{standard_metric}",
                keys=keys,
                config=config,
                annot=annot_lo_hi,
            )

            exit(-1)

            save_correlation_plot(
                data=corrmat,
                title=f"single_hyperparameter/{target_to_evaluate}/single_hyper_{target_to_evaluate}_{standard_metric}",
                keys=keys,
                config=config,
            )
