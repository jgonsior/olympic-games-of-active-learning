import csv
from datetime import timedelta
import glob
import multiprocessing
import sys
from typing import Any, Callable, Dict, List, Optional
from jinja2 import Template
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import Path
from timeit import default_timer as timer
from datasets import DATASET
from misc.config import Config
from misc.plotting import set_matplotlib_size, set_seaborn_style
import seaborn as sns

from resources.data_types import AL_STRATEGY


def append_and_create(file_name: Path, content: Dict):
    if not file_name.exists():
        with open(file_name, "w") as f:
            w = csv.DictWriter(f, fieldnames=content.keys())
            w.writeheader()

    with open(file_name, "a") as f:
        w = csv.DictWriter(f, fieldnames=content.keys())
        w.writerow(content)


def append_and_create_manually(file_name: Path, content: str):
    if not file_name.exists():
        file_name.parent.mkdir(parents=True, exist_ok=True)
        file_name.touch(exist_ok=True)

    with open(file_name, "a") as f:
        f.write(content)


# read in csv
# in case of errors -> skip file and return None
def get_df(file_name: Path, config: Config) -> Optional[pd.DataFrame]:
    # print(file_name)
    try:
        if file_name.name.endswith(".csv.xz") or file_name.name.endswith(".csv"):
            df = pd.read_csv(file_name)
        else:
            df = pd.read_parquet(file_name)
    except Exception as err:
        error_message = f"ERROR: {err.__class__.__name__} - {err.args}"
        print(error_message)
        print(err)

        append_and_create(
            config.BROKEN_CSV_FILE_PATH,
            {"metric_file": file_name, "error_message": error_message},
        )

        return None

    return df


def get_glob_list(
    config: Config,
    limit: str = "**/*",
    ignore_original_workloads=True,
) -> List[Path]:
    glob_list = [
        *[
            ggg
            for ggg in glob.glob(
                str(config.OUTPUT_PATH) + f"/{limit}.csv.xz", recursive=True
            )
        ],
        *[
            ggg
            for ggg in glob.glob(
                str(config.OUTPUT_PATH) + f"/{limit}.csv.xz.parquet", recursive=True
            )
        ],
    ]

    if ignore_original_workloads:
        glob_list = [
            ggg
            for ggg in glob_list
            if not ggg.endswith("_workload.csv.xz")
            and not ggg.endswith("_workloads.csv.xz")
        ]

    glob_list = [Path(ggg) for ggg in glob_list]
    return sorted(glob_list)


def get_done_workload_joined_with_metric(
    metric_name: str, config: Config
) -> pd.DataFrame:
    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

    del done_workload_df["EXP_RANDOM_SEED"]
    del done_workload_df["EXP_NUM_QUERIES"]

    def _do_stuff(file_name: Path, config: Config, done_workload_df: pd.DataFrame):
        print(file_name)

        metric_name = file_name.name.removesuffix(".parquet").removesuffix(".csv.xz")

        metric_df = get_df(file_name, config)

        if metric_df is None:
            return

        metric_df = pd.merge(
            metric_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left"
        )

        return metric_df

    glob_list = get_glob_list(config, limit=f"**/{metric_name}")

    # metric_dfs = Parallel(n_jobs=1, verbose=10)(
    metric_dfs = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config, done_workload_df)
        for file_name in glob_list
    )

    df = pd.concat(metric_dfs).reset_index(drop=True)

    return df


def get_done_workload_joined_with_multiple_metrics(
    metric_names: List[str], config: Config
) -> pd.DataFrame:
    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

    del done_workload_df["EXP_RANDOM_SEED"]
    del done_workload_df["EXP_NUM_QUERIES"]

    def _do_stuff(file_name: Path, config: Config, done_workload_df: pd.DataFrame):
        print(file_name)

        metric_name = file_name.name.removesuffix(".parquet").removesuffix(".csv.xz")

        metric_df = get_df(file_name, config)

        if metric_df is None:
            return

        metric_columns = [mmm for mmm in metric_df.columns]
        metric_columns.remove("EXP_UNIQUE_ID")

        metric_df[metric_columns] = metric_df[metric_columns].apply(
            pd.to_numeric, downcast="float"
        )

        metric_df["metric_name"] = pd.Series(
            [metric_name for _ in range(len(metric_df.index))]
        )

        metric_df = pd.merge(
            metric_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left"
        )

        return metric_df

    glob_list = []
    for metric_name in metric_names:
        glob_list = [*glob_list, *get_glob_list(config, limit=f"**/{metric_name}")]
    glob_list = set(glob_list)

    # metric_dfs = Parallel(n_jobs=1, verbose=10)(
    metric_dfs = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config, done_workload_df)
        for file_name in glob_list
    )

    df = pd.concat(metric_dfs).reset_index(drop=True)

    return df


def create_fingerprint_joined_timeseries_csv_files(
    metric_names: List[str], config: Config
):
    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

    del done_workload_df["EXP_RANDOM_SEED"]
    del done_workload_df["EXP_NUM_QUERIES"]

    def _do_stuff(file_name: Path, config: Config, done_workload_df: pd.DataFrame):
        # print(file_name)

        metric_name = file_name.name.removesuffix(".parquet").removesuffix(".csv.xz")

        metric_df = get_df(file_name, config)

        if metric_df is None:
            return

        metric_columns = [mmm for mmm in metric_df.columns]
        metric_columns.remove("EXP_UNIQUE_ID")

        metric_df[metric_columns] = metric_df[metric_columns].apply(
            pd.to_numeric, downcast="float"
        )

        metric_df = pd.merge(
            metric_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left"
        )

        ts_file = config.CORRELATION_TS_PATH / f"{metric_name}.unsorted.csv"

        contents = ""
        for _, row in metric_df.iterrows():
            row = row.to_list()

            non_metric_values = [str(int(rrr)) for rrr in row[-7:]]
            non_metric_values = [non_metric_values[0], ",".join(non_metric_values[1:])]

            for ix, v in enumerate(row[:-7]):
                if np.isnan(v):
                    continue
                contents += (
                    f"{non_metric_values[1]},{ix},{non_metric_values[0]}_{ix},{v}\n"
                )

        append_and_create_manually(ts_file, contents)

    glob_list = []
    for metric_name in metric_names:
        glob_list = [*glob_list, *get_glob_list(config, limit=f"**/{metric_name}")]
    glob_list = sorted(set(glob_list))

    # remove those from glob list which already exist as timeseries

    print(len(glob_list))

    existent_ts_files = [
        ggg.split("/")[-1].split(".")[0] + ".csv.xz"
        for ggg in glob.glob(
            str(config.CORRELATION_TS_PATH) + f"/*.parquet", recursive=True
        )
    ]

    glob_list = [ggg for ggg in glob_list if ggg.name not in existent_ts_files]
    print(len(glob_list))

    # metric_dfs = Parallel(n_jobs=1, verbose=10)(
    Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config, done_workload_df)
        for file_name in glob_list
    )


def save_correlation_plot(
    data: np.ndarray, title: str, keys: List[str], config: Config, total=False
):
    if title == "EXP_STRATEGY":
        keys = [AL_STRATEGY(int(kkk)).name for kkk in keys]
    elif title == "EXP_DATASET":
        keys = [DATASET(int(kkk)).name for kkk in keys]

    result_folder = Path(config.OUTPUT_PATH / f"plots/")
    result_folder.mkdir(parents=True, exist_ok=True)

    data_df = pd.DataFrame(data=data, columns=keys, index=keys)

    data_df.to_parquet(result_folder / f"{title}.parquet")

    if total:
        data_df.loc[:, "Total"] = data_df.mean(axis=1)
        data_df.sort_values(by=["Total"], inplace=True)

    print(data_df)
    print(result_folder / f"{title}.jpg")
    set_seaborn_style(font_size=8)
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    plt.figure(figsize=set_matplotlib_size(fraction=len(keys) / 12))
    ax = sns.heatmap(data_df, annot=True, fmt=".2f")

    ax.set_title(title)

    plt.savefig(
        result_folder / f"{title}.jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )


def log_and_time(log_message: str):
    if not hasattr(log_and_time, "last_time"):
        log_and_time.last_time = timer()
    now = timer()
    print(f"{timedelta(seconds=now - log_and_time.last_time)}: {log_message}")

    log_and_time.last_time = now


def create_workload(
    workload: List[Any],
    config: Config,
    SLURM_ITERATIONS_PER_BATCH: int,
    SCRIPTS_PATH: str,
    SLURM_NR_THREADS: int,
    CLI_ARGS: str = "",
    # SLURM_MEMORY: int,
):
    df = pd.DataFrame(workload)

    # if DONE exists only rerun those who are actually new
    if config.EVA_SCRIPT_DONE_WORKLOAD_FILE.exists():
        done_df = pd.read_csv(
            config.EVA_SCRIPT_DONE_WORKLOAD_FILE,
            index_col=None,
            header=0,
            usecols=[iii for iii in range(0, len(workload[0]))],
        )

        for _, r in done_df.iterrows():

            mask = True
            for k, v in enumerate(r.to_list()):
                mask &= df[k] == v
            df = df.loc[~mask]

    df.to_csv(config.EVA_SCRIPT_OPEN_WORKLOAD_FILE, index=False)
    print(f"Created workload of {len(df)}")

    # create slurm file
    slurm_template = Template(
        Path("resources/slurm_templates/eva_parallel.sh").read_text()
    )
    rendered_template = slurm_template.render(
        SLURM_TIME_LIMIT="100:00:00",
        SLURM_NR_THREADS=SLURM_NR_THREADS,
        # SLURM_MEMORY=SLURM_MEMORY,
        EVA_SCRIPT_WORKLOAD_DIR=config.EVA_SCRIPT_WORKLOAD_DIR.parent.name,
        EVA_NAME=config.EVA_NAME,
        START=0,
        END=int(len(df) / SLURM_ITERATIONS_PER_BATCH),
        SLURM_OFFSET=0,
        SLURM_ITERATIONS_PER_BATCH=SLURM_ITERATIONS_PER_BATCH,
        PYTHON_FILE=Path(sys.argv[0]).name.removeprefix(".py"),
        HPC_OUTPUT_PATH=config.HPC_OUTPUT_PATH,
        HPC_WS_PATH=config.HPC_WS_PATH,
        HPC_PYTHON_PATH=config.HPC_PYTHON_PATH,
        EXP_TITLE=config.EXP_TITLE,
        str=str,
        SCRIPTS_PATH=SCRIPTS_PATH,
        HPC_SLURM_PROJECT=config.HPC_SLURM_PROJECT,
        CLI_ARGS=CLI_ARGS,
    )

    print(rendered_template)

    Path(config.EVA_SCRIPT_WORKLOAD_DIR / "02_slurm.slurm").write_text(
        rendered_template
    )


def run_from_workload(do_stuff: Callable, config: Config):
    if config.EVA_MODE == "local":
        skip_rows = None
    else:
        skip_rows = lambda xxx: xxx not in [0, config.WORKER_INDEX + 1]

    workload_df = pd.read_csv(
        config.EVA_SCRIPT_OPEN_WORKLOAD_FILE,
        header=0,
        index_col=None,
        skiprows=skip_rows,
    )

    def do_stuff_wrapper(*args, do_stuff, config: Config):
        res = {kkk: vvv for kkk, vvv in enumerate(args)}
        res["result"] = do_stuff(*args, config)

        append_and_create(config.EVA_SCRIPT_DONE_WORKLOAD_FILE, res)

    if config.EVA_MODE == "local":
        Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
            delayed(do_stuff_wrapper)(*wl, do_stuff=do_stuff, config=config)
            for wl in workload_df.to_numpy().tolist()
        )
    elif config.EVA_MODE == "single" or config.EVA_MODE == "slurm":
        do_stuff_wrapper(
            *workload_df.loc[0].to_list(), do_stuff=do_stuff, config=config
        )


def _combine_results(config: Config) -> pd.DataFrame:
    return
