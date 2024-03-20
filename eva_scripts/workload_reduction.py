import csv
from pathlib import Path
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing

from misc.helpers import (
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
    save_correlation_plot,
)
from scipy.stats import pearsonr

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

prepare_eva_pathes("workload_reduction", config)


standard_metric = "full_auc_macro_f1-score"
#  standard_metric = "macro_f1-score"

if config.EVA_MODE == "create":
    if Path(config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv").exists():
        with open(Path(config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv"), "r") as f:
            length = sum(1 for _ in f)
    else:
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

        from pandarallel import pandarallel

        pandarallel.initialize(
            nb_workers=multiprocessing.cpu_count(),
            progress_bar=True,
            use_memory_fs=True,
        )

        ts["fingerprint"] = ts[fingerprint_cols].parallel_apply(
            lambda row: "_".join([str(rrr) for rrr in row]), axis=1
        )

        for fg_col in fingerprint_cols:
            del ts[fg_col]

        ts = ts.pivot(
            index="fingerprint", columns="EXP_STRATEGY", values="metric_value"
        ).reset_index()
        print(ts)

        ts.to_csv(config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv", index=None)

        length = len(ts)
        del ts

    print("done reading")

    workload = []
    for iii in range(1, length - 2, 2):
        workload.append([iii, iii + 1])

    create_workload(
        workload,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1000,
        SCRIPTS_PATH="scripts",
        SLURM_NR_THREADS=128,
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:

    def do_stuff(fingerprint1, fingerprint2, config):
        fingerprint1 = int(fingerprint1) + 1
        fingerprint2 = int(fingerprint2) + 1

        ts = pd.read_csv(
            config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv",
            header=0,
            skiprows=lambda rrr: rrr not in [0, fingerprint1, fingerprint2],
        ).dropna(axis=1)

        cols_without_fingerprint = list(ts.columns)
        cols_without_fingerprint.remove("fingerprint")

        stat = pearsonr(
            ts.iloc[0][cols_without_fingerprint].to_numpy(),
            ts.iloc[1][cols_without_fingerprint].to_numpy(),
        )[0]

        return stat

    run_from_workload(do_stuff=do_stuff, config=config)
elif config.EVA_MODE == "combine":
    ts_df = pd.read_csv(
        config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv", header=0, usecols=[0]
    )

    done_df = pd.read_csv(config.EVA_SCRIPT_DONE_WORKLOAD_FILE, header=0)
    done_df = pd.read_csv(config.EVA_SCRIPT_WORKLOAD_DIR / "03_done_01.csv", header=0)
    done_df.dropna(inplace=True)

    counts, bins = np.histogram(done_df["result"].to_numpy(), bins=200)

    for c, b in zip(counts, bins):
        print(f"{b}:\t{c}")
    plt.hist(bins[:-1], bins, weights=counts)
    plt.show()
    print("WIP")
    # now I need to decide which correlation qualifie as "dense" and which don't
    exit(0)
    corrmatt = np.ones(shape=(len(ts_df), len(ts_df)))

    with open(config.EVA_SCRIPT_DONE_WORKLOAD_FILE) as f:
        reader = csv.reader(f)

        # skip first row
        next(reader)
        for row in reader:
            ix1 = int(row[0])
            ix2 = int(row[1])
            value = float(row[2])

            corrmatt[ix1, ix2] = value
            corrmatt[ix2, ix1] = value

    print(corrmatt)

    save_correlation_plot(
        data=corrmatt,
        title="Necessary Workload",
        keys=ts_df["fingerprint"].to_list(),
        total=True,
        config=config,
    )
