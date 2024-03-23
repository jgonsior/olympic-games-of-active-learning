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
from scipy.stats import pearsonr, kendalltau

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

prepare_eva_pathes("workload_reduction", config)


standard_metric = "full_auc_macro_f1-score"
standard_metric = "last_5_macro_f1-score"
# standard_metric = "macro_f1-score"

if config.EVA_MODE == "create":
    ix_path = Path(config.EVA_SCRIPT_WORKLOAD_DIR / f"ts_ix_{config.WORKER_INDEX}.csv")
    if Path(config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv").exists():
        ix_df = pd.read_csv(ix_path)
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
        ix_df = pd.DataFrame(ts.index)
        ix_df.rename(columns={0: "0"}, inplace=True)

        ix_df.to_csv(ix_path, index=None)

        del ts

    print("done reading")
    ix_df["1"] = ix_df["0"][1:].reset_index()["0"]
    ix_df = ix_df[:-1]
    ix_df = ix_df.iloc[::2]
    ix_df["1"] = ix_df["1"].astype(int)

    if config.WORKER_INDEX > 0:
        last_done_df = pd.read_parquet(
            config.EVA_SCRIPT_WORKLOAD_DIR / f"03_done_{config.WORKER_INDEX-1}.parquet"
        )

        ix_df = (
            pd.merge(ix_df, last_done_df[["0", "1"]], indicator=True, how="outer")
            .query('_merge=="left_only"')
            .drop("_merge", axis=1)
        )
    ix_df.to_csv(config.EVA_SCRIPT_OPEN_WORKLOAD_FILE, index=False)
    exit(-1)
    workload = []
    last_one = False
    for iii, jjj in zip(ix_df["0"][:-1], ix_df["0"][1:]):
        if last_one:
            last_one = False
            continue

        if config.WORKER_INDEX > 0:
            if (
                len(
                    last_done_df.loc[
                        (last_done_df["0"] == iii) & (last_done_df["1"] == jjj)
                    ]
                )
                == 0
            ):
                workload.append([iii, jjj])
        else:
            workload.append([iii, jjj])

    print("done removing old stuff")

    exit(-1)

    create_workload(
        workload,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1000,
        SCRIPTS_PATH="scripts",
        SLURM_NR_THREADS=128,
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:
    from pandarallel import pandarallel

    pandarallel.initialize(
        nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
    )
    workload_df = pd.read_csv(
        config.EVA_SCRIPT_OPEN_WORKLOAD_FILE,
        header=0,
        index_col=None,
    )
    workload_df.rename(columns={"0": "ix", "1": "ix2"}, inplace=True)

    # enhance by fingerprints
    ts = pd.read_csv(
        config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv",
        header=0,
    )
    del ts["fingerprint"]

    ts["ts_ix"] = ts.index

    workload_df = pd.merge(
        left=workload_df, right=ts, left_on="ix", right_on="ts_ix", how="left"
    )
    workload_df = pd.merge(
        left=workload_df, right=ts, left_on="ix2", right_on="ts_ix", how="left"
    )

    del workload_df["ts_ix_x"]
    del workload_df["ts_ix_y"]

    def do_stuff_pearson(row: pd.Series):
        a = np.array([rrr[1] for rrr in row.items() if rrr[0].endswith("_x")])
        b = np.array([rrr[1] for rrr in row.items() if rrr[0].endswith("_y")])

        bad = ~np.logical_or(np.isnan(a), np.isnan(b))
        a = a[bad]
        b = b[bad]
        stat = pearsonr(a, b)

        return pd.Series([stat.statistic, stat.pvalue])

    def do_stuff(row: pd.Series):
        a = np.array([rrr[1] for rrr in row.items() if rrr[0].endswith("_x")])
        b = np.array([rrr[1] for rrr in row.items() if rrr[0].endswith("_y")])

        bad = ~np.logical_or(np.isnan(a), np.isnan(b))
        a = a[bad]
        b = b[bad]

        stat = kendalltau(a, b, nan_policy="omit")

        return pd.Series([stat.statistic, stat.pvalue])

    # print(workload_df)
    result = pd.DataFrame(workload_df.parallel_apply(do_stuff, axis=1))
    result.rename(columns={0: "result_0", 1: "result_1"}, inplace=True)

    result["0"] = workload_df["ix"]
    result["1"] = workload_df["ix2"]
    result.to_csv(config.EVA_SCRIPT_DONE_WORKLOAD_FILE, index=False)
    exit(-1)

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
        )

        return stat.statistic, stat.pvalue

    run_from_workload(do_stuff=do_stuff, config=config, return_list=True)
elif config.EVA_MODE == "analyze_plot":
    ts_df = pd.read_csv(
        config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv", header=0, usecols=[0]
    )

    done_df = pd.read_parquet(
        config.EVA_SCRIPT_WORKLOAD_DIR / f"03_done_{config.WORKER_INDEX}.parquet",
    )
    # done_df = pd.read_csv(config.EVA_SCRIPT_DONE_WORKLOAD_FILE)
    print(done_df)
    done_df.sort_values(by=["0", "1"], inplace=True)
    done_df.to_csv("test.csv", index=False)
    exit(-1)
    done_df.dropna(inplace=True)

    counts, bins = np.histogram(done_df["result_1"].to_numpy(), bins=200)

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

elif config.EVA_MODE == "reduce":
    ts_df = pd.read_csv(
        config.EVA_SCRIPT_WORKLOAD_DIR / "ts.csv", header=0, usecols=[0]
    )
    ix_path = Path(config.EVA_SCRIPT_WORKLOAD_DIR / f"ts_ix_{config.WORKER_INDEX}.csv")
    new_ix_path = Path(
        config.EVA_SCRIPT_WORKLOAD_DIR / f"ts_ix_{config.WORKER_INDEX+1}.csv"
    )

    try:
        ori_done_df = pd.read_csv(config.EVA_SCRIPT_DONE_WORKLOAD_FILE, header=0)
        ori_done_df.dropna(inplace=True)

        # we keep all those where we don't have a correlation
        done_df = ori_done_df.loc[
            (ori_done_df["result_0"] > config.EVA_WORKLOAD_REDUCTION_THRESHOLD)
            & ((ori_done_df["result_1"] < 0.05))
        ]

        ts_ix = pd.read_csv(ix_path)
        ts_ix = ts_ix.loc[~ts_ix["0"].isin(done_df["1"])]

    except:
        ori_done_df = pd.read_parquet(
            config.EVA_SCRIPT_WORKLOAD_DIR / f"03_done_{config.WORKER_INDEX-1}.parquet",
        )
        ori_done_df = ori_done_df[0:0]

        ts_ix = pd.read_csv(ix_path)

    if config.WORKER_INDEX > 0:
        last_ts_ix = pd.read_csv(
            config.EVA_SCRIPT_WORKLOAD_DIR / f"ts_ix_{config.WORKER_INDEX-1}.csv"
        )

        if len(last_ts_ix) == len(ts_ix):
            ts_ix = ts_ix.sample(frac=1)

    ts_ix.to_csv(new_ix_path, index=False)

    archived_done_path = (
        config.EVA_SCRIPT_WORKLOAD_DIR / f"03_done_{config.WORKER_INDEX}.parquet"
    )
    print(len(ori_done_df))
    if config.WORKER_INDEX > 0:
        for i in range(0, config.WORKER_INDEX):
            last_last_done_df = pd.read_parquet(
                config.EVA_SCRIPT_WORKLOAD_DIR / f"03_done_{i}.parquet"
            )

            ori_done_df = pd.concat([last_last_done_df, ori_done_df])
            ori_done_df.drop_duplicates(inplace=True)
    print(len(ori_done_df))
    try:
        config.EVA_SCRIPT_DONE_WORKLOAD_FILE.unlink()
    except:
        print("hm")
    ori_done_df.to_parquet(archived_done_path)
