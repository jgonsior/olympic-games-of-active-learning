import ast
import csv
from datetime import timedelta
import glob
from itertools import combinations
import multiprocessing
import sys
from typing import Any, Callable, Dict, List, Optional
from jinja2 import Template
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from natsort import natsort_keygen, natsorted
import numpy as np
import pandas as pd
from pathlib import Path
from timeit import default_timer as timer

import scipy
import scipy.stats
from datasets import DATASET
from misc.config import Config
from misc.plotting import (
    _rename_learner_model,
    _rename_strategy,
    set_matplotlib_size,
    set_seaborn_style,
)
import seaborn as sns

from resources.data_types import AL_STRATEGY, LEARNER_MODEL


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
    print(str(config.OUTPUT_PATH) + f"/{limit}.csv.xz")
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
    metric_names: List[str],
    config: Config,
):
    done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

    del done_workload_df["EXP_RANDOM_SEED"]
    del done_workload_df["EXP_NUM_QUERIES"]

    def _join_al_cycles_to_auc_like(row):
        result = ""
        for r in row:
            if pd.notna(r):
                result = result + ", " + str(r).removeprefix("[").removesuffix("]")

        return "[" + result.removeprefix(", ").replace(",", "") + "]"

    def _do_stuff(file_name: Path, config: Config, done_workload_df: pd.DataFrame):
        # print(file_name)

        metric_name = file_name.name.removesuffix(".parquet").removesuffix(".csv.xz")

        metric_df = get_df(file_name, config)

        if metric_df is None:
            return

        metric_columns = [mmm for mmm in metric_df.columns]
        metric_columns.remove("EXP_UNIQUE_ID")

        if "selected_indices" != metric_name:
            metric_df[metric_columns] = metric_df[metric_columns].apply(
                pd.to_numeric, downcast="float"
            )
        else:
            metric_df.insert(
                0,
                "selected_indices",
                metric_df[metric_columns].apply(_join_al_cycles_to_auc_like, axis=1),
            )

            for metric_column in metric_columns:
                del metric_df[metric_column]

        metric_df = pd.merge(
            metric_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left"
        )

        ts_file = config.CORRELATION_TS_PATH / f"{metric_name}.unsorted.csv"

        contents = ""
        for _, row in metric_df.iterrows():
            non_metric_values = [str(int(rrr)) for rrr in row[-7:]]
            non_metric_values = [
                non_metric_values[0],
                ",".join(non_metric_values[1:]),
            ]

            for ix, v in enumerate(row[:-7]):
                if metric_name == "selected_indices" and str(v) == "nan":
                    continue
                elif metric_name != "selected_indices" and np.isnan(v):
                    continue

                contents += (
                    f"{non_metric_values[1]},{ix},{non_metric_values[0]}_{ix},{v}\n"
                )
        append_and_create_manually(ts_file, contents)

    glob_list = []
    for metric_name in metric_names:
        glob_list = [*glob_list, *get_glob_list(config, limit=f"**/{metric_name}")]
    glob_list = sorted(set(glob_list))
    #  print(glob_list[:10])
    #  print(glob_list[-10:])

    # remove those from glob list which already exist as timeseries

    #  print(len(glob_list))

    existent_ts_files = [
        ggg.split("/")[-1].split(".")[0] + ".csv.xz"
        for ggg in glob.glob(
            str(config.CORRELATION_TS_PATH) + f"/*.parquet", recursive=True
        )
    ]
    #  print(existent_ts_files[:10])

    glob_list = [ggg for ggg in glob_list if ggg.name not in existent_ts_files]
    print(len(glob_list))

    # Parallel(n_jobs=1, verbose=10)(
    Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config, done_workload_df)
        for file_name in glob_list
    )


def _calculate_fig_size(fig_width_in, heigh_bonus=1):
    golden_ratio = (5**0.5 - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * heigh_bonus
    return (fig_width_in, fig_height_in)


def save_correlation_plot(
    data: np.ndarray,
    title: str,
    keys: List[str],
    config: Config,
    total=False,
    rotation=False,
):

    # keys = [kkk.removesuffix("_weighted_f1-score") for kkk in keys]

    if "standard_metric_kendall" in title:
        keys = [kkk.removeprefix("full_auc_") for kkk in keys]
        keys = [kkk.replace("Full Mean", "Class Weighted F1-Score") for kkk in keys]
    elif "AUC/auc_weighted_f1-score" in title:
        keys = [kkk.replace("Class Weighted F1-Score", "full_auc") for kkk in keys]

        if not "First 5" in keys:
            keys = [f"{kkk}_weighted_f1-score" for kkk in keys]
    elif "leaderboard_types_kendall" in title:
        keys = [kkk.replace("_full_auc_weighted_f1-score", "") for kkk in keys]
        keys = [
            kkk.replace(
                "dataset_normalized_percentages_", "Percentages Dataset Normalization"
            )
            for kkk in keys
        ]
        keys = [kkk.replace("percentages_", "Percentages") for kkk in keys]
        keys = [kkk.replace("rank_", "Ranks") for kkk in keys]
        keys = [kkk.replace("dense_none", " Dense") for kkk in keys]
        keys = [kkk.replace("sparse_none", " Sparse None") for kkk in keys]
        keys = [kkk.replace("sparse_zero", " Sparse Zero") for kkk in keys]
    renaming_dict = {
        "accuracy": "Accuracy",
        "weighted_f1-score": "Class Weighted F1-Score",
        "weighted_precision": "Class Weighted Precision",
        "weighted_recall": "Class Weighted Recall",
        "macro_f1-score": "Macro F1-Score",
        "macro_recall": "Macro Recall",
        "macro_precision": "Macro Precision",
        "ramp_up_auc_weighted_f1-score": "Ramp-Up",
        "first_5_weighted_f1-score": "First 5",
        "final_value_weighted_f1-score": "Last Value",
        "last_5_weighted_f1-score": "Last 5",
        "plateau_auc_weighted_f1-score": "Plateau",
        "full_auc_weighted_f1-score": "Full Mean",
        "gold standard": "ZGold Standard",
    }

    keys = [renaming_dict[kkk] if kkk in renaming_dict else kkk for kkk in keys]

    if "EXP_STRATEGY" in title:
        rotation = 45
        try:
            keys = [_rename_strategy(AL_STRATEGY(int(kkk)).name) for kkk in keys]
        except:
            keys = keys
    elif "EXP_DATASET" in title:
        try:
            keys = [DATASET(int(kkk)).name for kkk in keys]
        except:
            keys = keys
    elif "EXP_LEARNER_MODEL" in title:
        # try:
        keys = [
            _rename_learner_model(kkk) if kkk is not "Gold Standard" else kkk
            for kkk in keys
        ]
        # except:
        keys = keys

    result_folder = Path(config.OUTPUT_PATH / f"plots/")
    result_folder.mkdir(parents=True, exist_ok=True)

    data_df = pd.DataFrame(data=data, columns=keys, index=keys)
    data_df = data_df.sort_index(axis=0)
    data_df = data_df.sort_index(axis=1)

    data_df = data_df.rename({"ZGold Standard": "Gold Standard"}, axis=0)
    data_df = data_df.rename({"ZGold Standard": "Gold Standard"}, axis=1)
    Path(result_folder / f"{title}").parent.mkdir(exist_ok=True, parents=True)
    data_df.to_parquet(result_folder / f"{title}.parquet")

    if total:
        data_df.loc[:, "Total"] = data_df.mean(axis=1)
        # data_df.sort_values(by=["Total"], inplace=True)

    print(data_df)
    print(result_folder / f"{title}.jpg")
    set_seaborn_style(font_size=5)
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys

    if "Standard Metrics" in title:
        figsize = _calculate_fig_size(3)
    elif "AUC/auc_weighted_f1-score" in title:
        # set_seaborn_style(font_size=6)
        figsize = _calculate_fig_size(3)
        rotation = 30
    elif "standard_metric_kendall" in title:
        figsize = _calculate_fig_size(3)
    elif "auc_metric_kendall" in title:
        figsize = _calculate_fig_size(3)
    elif "leaderboard_types_kendall" in title:
        figsize = _calculate_fig_size(0.5 * 7.1413)
        rotation = 30
        set_seaborn_style(font_size=6)
    elif "single_indice_EXP_BATCH_SIZE_full_auc__selected_indices_jaccard" in title:
        figsize = _calculate_fig_size(2)
    elif "single_hyper_EXP_BATCH_SIZE_full_auc_weighted_f1-score" in title:
        figsize = _calculate_fig_size(2)
        # set_seaborn_style(font_size=5)
    elif "EXP_BATCH_SIZE_kendall" in title:
        figsize = _calculate_fig_size(2)

        custom_dict = {"1": 1, "10": 10, "100": 100, "20": 20, "5": 5, "50": 50}

        data_df = data_df.sort_index(axis=0, key=lambda x: x.map(custom_dict))
        data_df = data_df.sort_index(axis=1, key=lambda x: x.map(custom_dict))

        rotation = 30
    elif "single_hyper_EXP_DATASET_full_auc_weighted_f1" in title:
        figsize = _calculate_fig_size(1 * 7.1413)
        set_seaborn_style(font_size=6)
    elif "single_hyper_EXP_LEARNER_MODEL_full_auc_weighted_f1" in title:
        figsize = _calculate_fig_size(1.5)
        set_seaborn_style(font_size=7)
        rotation = 30
    elif "single_indice_EXP_LEARNER_MODEL_full_auc__selected_indices_jaccard" in title:
        figsize = _calculate_fig_size(1.5)
        set_seaborn_style(font_size=7)
        rotation = 30
    elif "EXP_LEARNER_MODEL_kendall" in title:
        figsize = _calculate_fig_size(1.5)
        set_seaborn_style(font_size=7)
        rotation = 30
    elif "single_hyper_EXP_STRATEGY_full_auc_weighted_f1" in title:
        figsize = _calculate_fig_size(0.92 * 7.1413)
        set_seaborn_style(font_size=4)
    elif "single_indice_EXP_STRATEGY_full_auc__selected_indices_jaccard" in title:
        figsize = _calculate_fig_size(0.92 * 7.1413)
        set_seaborn_style(font_size=4)
    elif "EXP_TRAIN_TEST_BUCKET_SIZE_kendall" in title:
        figsize = _calculate_fig_size(2.5)
        # set_seaborn_style(font_size=7)
        rotation = 30
    elif "EXP_START_POINT_kendall" in title:
        figsize = _calculate_fig_size(0.92 * 7.1413)
        set_seaborn_style(font_size=5)
        # set_seaborn_style(font_size=7)

        custom_dict = {str(kkk): kkk for kkk in range(0, 20)}

        data_df = data_df.sort_index(axis=0, key=lambda x: x.map(custom_dict))
        data_df = data_df.sort_index(axis=1, key=lambda x: x.map(custom_dict))
        rotation = 30
    else:
        figsize = set_matplotlib_size(fraction=len(keys) / 12)
    plt.figure(figsize=figsize)

    data_df = data_df * 100
    ax = sns.heatmap(
        data_df,
        annot=True,
        fmt=".1f",
        vmin=0,
        vmax=100,
        cmap=sns.color_palette("flare_r", as_cmap=True),
        # square=True,
    )
    # ax = sns.heatmap(data_df, annot=True, fmt=".2%", vmin=0, vmax=1)
    ax.tick_params(left=False, bottom=False, pad=-4)

    ax.set_title("")

    if rotation is not False:
        # plt.xticks(rotation=rotation)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right")
        # ax.tick_params(axis="x", labelrotation=rotation)

    plt.savefig(
        result_folder / f"{title}.jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.savefig(
        result_folder / f"{title}.pdf",
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
    script_type: str = "script",
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
    print(config.EVA_NAME)
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
        PYTHON_FILE=Path(sys.argv[0]).name.removesuffix(".py"),
        HPC_OUTPUT_PATH=config.HPC_OUTPUT_PATH,
        HPC_WS_PATH=config.HPC_WS_PATH,
        HPC_PYTHON_PATH=config.HPC_PYTHON_PATH,
        EXP_TITLE=config.EXP_TITLE,
        str=str,
        SCRIPTS_PATH=SCRIPTS_PATH,
        HPC_SLURM_PROJECT=config.HPC_SLURM_PROJECT,
        CLI_ARGS=CLI_ARGS,
        script_type=script_type,
    )

    print(rendered_template)

    Path(config.EVA_SCRIPT_WORKLOAD_DIR / "02_slurm.slurm").write_text(
        rendered_template
    )


def run_from_workload(do_stuff: Callable, config: Config, return_list=False):
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

    def do_stuff_wrapper(*args, do_stuff, return_list, config: Config):
        #  try:
        res = {kkk: vvv for kkk, vvv in enumerate(args)}
        res["result"] = do_stuff(*args, config)

        if return_list:
            for ix, rrr in enumerate(res["result"]):
                res[f"result_{ix}"] = rrr
            del res["result"]

        print("WORKLOAD JOB DONE")
        append_and_create(config.EVA_SCRIPT_DONE_WORKLOAD_FILE, res)
        #  except:
        #  print("ERROR" * 100)

    if config.EVA_MODE == "local":
        Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
            delayed(do_stuff_wrapper)(
                *wl, do_stuff=do_stuff, config=config, return_list=return_list
            )
            for wl in workload_df.to_numpy().tolist()
        )
    elif config.EVA_MODE == "single" or config.EVA_MODE == "slurm":
        do_stuff_wrapper(
            *workload_df.loc[0].to_list(),
            do_stuff=do_stuff,
            config=config,
            return_list=return_list,
        )


def prepare_eva_pathes(eva_name: str, config: Config):
    config.EVA_NAME = eva_name
    config.EVA_SCRIPT_WORKLOAD_DIR = config.EVA_SCRIPT_WORKLOAD_DIR / config.EVA_NAME
    config.EVA_SCRIPT_OPEN_WORKLOAD_FILE = (
        config.EVA_SCRIPT_WORKLOAD_DIR / config.EVA_SCRIPT_OPEN_WORKLOAD_FILE
    )
    config.EVA_SCRIPT_DONE_WORKLOAD_FILE = (
        config.EVA_SCRIPT_WORKLOAD_DIR / config.EVA_SCRIPT_DONE_WORKLOAD_FILE
    )

    if not config.EVA_SCRIPT_WORKLOAD_DIR.exists():
        config.EVA_SCRIPT_WORKLOAD_DIR.mkdir(parents=True)


def _flatten(xss):
    return [x for xs in xss for x in xs]


def parallel_correlation(df: pd.DataFrame, method: str) -> np.ndarray:
    def _do_stuff(col1, col2, df):
        kendalltau = scipy.stats.kendalltau(df[col1], df[col2])

        res = np.nan
        # if kendalltau.pvalue < 0.05:
        res = kendalltau.statistic
        return [(col1, col2, res), (col2, col1, res)]

    wl = list(combinations(df.columns, 2))
    print(len(wl))
    results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(col1, col2, df) for col1, col2 in wl
    )
    results = pd.DataFrame(_flatten(results), columns=["x", "y", "stat"]).pivot(
        index="x", columns="y", values="stat"
    )

    return results
