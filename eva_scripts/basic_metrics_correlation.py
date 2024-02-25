import itertools
from pathlib import Path
import sys
import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import pandas as pd
from misc.plotting import set_seaborn_style, set_matplotlib_size
import multiprocessing

sys.dont_write_bytecode = True

from misc.config import Config
import numpy as np
import seaborn as sns


config = Config()


def _is_standard_metric(metric_path: str, config: Config) -> bool:
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

    if (
        config.EVA_METRICS_TO_CORRELATE == "extended"
        or config.EVA_METRICS_TO_CORRELATE == "auc"
    ):

        variant_prefixe = [
            "biggest_drop_per_",
            "nr_decreasing_al_cycles_per_",
        ]

        original_standard_metrics = standard_metrics.copy()
        for vp in variant_prefixe:
            standard_metrics = [
                *standard_metrics,
                *[vp + sss for sss in original_standard_metrics],
            ]

        if config.EVA_METRICS_TO_CORRELATE == "auc":
            auc_prefixe = [
                "final_value_",
                "first_5_",
                "full_auc_",
                "last_5_",
                "learning_stability_5_",
                "learning_stability_10_",
                "ramp_up_auc_",
                "plateu_auc_",
            ]

            original_standard_metrics = standard_metrics.copy()

            for vp in auc_prefixe:
                standard_metrics = [
                    *standard_metrics,
                    *[vp + sss for sss in original_standard_metrics],
                ]

        standard_metrics = [
            *standard_metrics,
            *[sss + "_time_lag" for sss in standard_metrics],
        ]
    for sm in standard_metrics:
        if f"/{sm}.csv" in metric_path:
            return True
    return False


def _do_stuff(exp_dataset, exp_strategy, config):
    # if exp_strategy.name != "SMALLTEXT_PREDICTIONENTROPY":
    #    return
    glob_list = [
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH)
            + f"/{exp_strategy.name}/{exp_dataset.name}/*.csv.xz",
            recursive=True,
        )
        if _is_standard_metric(f, config)
    ]

    if len(glob_list) == 0:
        return

    metric_dfs = {}

    exp_ids = []
    for file_name in glob_list:
        # print(file_name)
        metric_name = Path(file_name).name.removesuffix(".csv.xz")
        metric_dfs[metric_name] = pd.read_csv(file_name).sort_values(by="EXP_UNIQUE_ID")

        if len(exp_ids) == 0:
            exp_ids = set(metric_dfs[metric_name]["EXP_UNIQUE_ID"].tolist())
        else:
            difference = set(
                metric_dfs[metric_name]["EXP_UNIQUE_ID"].tolist()
            ).symmetric_difference(exp_ids)
            if len(difference) != 0:
                # save exp_unique_id
                # delete them for now from all dfs to minimum
                print("exp_ids missing")
                print(difference)
                print(file_name)
                exit(-1)

    summed_up_corr_values = None

    for ix, row in list(metric_dfs.values())[0].iterrows():
        correlation_data = []

        for metric, metric_df in metric_dfs.items():
            correlation_data.append(
                [
                    metric,
                    *metric_df.iloc[ix].to_list()[:-1],
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


result_folder = Path(config.OUTPUT_PATH / f"plots/")
result_folder.mkdir(parents=True, exist_ok=True)

summed_up_corr_values.to_parquet(
    result_folder / f"{config.EVA_METRICS_TO_CORRELATE}.parquet"
)

summed_up_corr_values = summed_up_corr_values.map(lambda r: np.mean(r))
summed_up_corr_values.loc[:, "Total"] = summed_up_corr_values.mean(axis=1)
summed_up_corr_values.sort_values(by=["Total"], inplace=True)

print(summed_up_corr_values)

set_seaborn_style(font_size=8)
fig = plt.figure(figsize=set_matplotlib_size())
sns.heatmap(summed_up_corr_values, annot=True)

plt.savefig(
    result_folder / f"{config.EVA_METRICS_TO_CORRELATE}.jpg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
