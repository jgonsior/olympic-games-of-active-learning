from collections import defaultdict
import itertools
import multiprocessing
from pathlib import Path
import sys
import glob
import warnings
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from datasets import DATASET
from misc.plotting import set_seaborn_style, set_matplotlib_size

from resources.data_types import AL_STRATEGY

sys.dont_write_bytecode = True

from misc.config import Config
import numpy as np
import seaborn as sns

from scipy.stats import spearmanr


config = Config()


def _is_standard_metric(metric_path: str) -> bool:
    standard_metrics = [
        "accuracy",
        "weighted_recall",
        "macro_f1-score",
        "macro_precision",
        "macro_recall",
        "weighted_f1-score",
        "weighted_precision",
        "weighted_recall",
    ]

    for sm in standard_metrics:
        if f"{sm}.csv" in metric_path:
            return True
    return False


def _do_stuff(exp_dataset, exp_strategy, config):
    glob_list = [
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH)
            + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv.xz",
            recursive=True,
        )
        if _is_standard_metric(f)
    ]

    metric_dfs = {}
    for file_name in glob_list:
        metric_name = Path(file_name).name.removesuffix(".csv.xz")
        metric_dfs[metric_name] = pd.read_csv(file_name)

    if len(metric_dfs) == 0:
        return

    summed_up_corr_values = None

    for EXP_UNIQUE_ID in list(metric_dfs.values())[0]["EXP_UNIQUE_ID"]:

        correlation_data = []

        for metric, metric_df in metric_dfs.items():
            correlation_data.append(
                [
                    metric,
                    *metric_df.loc[metric_df["EXP_UNIQUE_ID"] == EXP_UNIQUE_ID]
                    .iloc[0]
                    .to_list()[:-1],
                ]
            )
        correlation_matrix = pd.DataFrame(correlation_data).T
        headers = correlation_matrix.iloc[0].values
        correlation_matrix.columns = headers
        correlation_matrix.drop(index=0, axis=0, inplace=True)
        correlation_matrix.dropna(how="all", inplace=True)

        corr_values = correlation_matrix.corr().map(lambda r: [r])

        if summed_up_corr_values is None:
            summed_up_corr_values = corr_values
        else:
            summed_up_corr_values = summed_up_corr_values + corr_values

    return summed_up_corr_values


# dfs = Parallel(n_jobs=1, verbose=10)(
dfs = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(exp_dataset, exp_strategy, config)
    for (exp_dataset, exp_strategy) in itertools.product(
        config.EXP_GRID_DATASET, config.EXP_GRID_STRATEGY
    )
)


summed_up_corr_values = None
for df in dfs:
    if df is None:
        continue
    if summed_up_corr_values is None:
        summed_up_corr_values = df
    else:
        summed_up_corr_values = summed_up_corr_values + df


result_folder = Path(f"plots/{config.EXP_TITLE}/")
result_folder.mkdir(parents=True, exist_ok=True)

summed_up_corr_values.to_csv(result_folder / "basic_metrics.csv")

summed_up_corr_values = summed_up_corr_values.map(lambda r: np.mean(r))

set_seaborn_style(font_size=8)
fig = plt.figure(figsize=set_matplotlib_size())
sns.heatmap(summed_up_corr_values, annot=True)

plt.savefig(
    result_folder / "basic_metrics.jpg", dpi=300, bbox_inches="tight", pad_inches=0
)
