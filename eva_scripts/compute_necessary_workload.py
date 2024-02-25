import ast
import copy
import csv
import itertools
from pathlib import Path
import sys
import glob
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing
import seaborn as sns
from sklearn.isotonic import spearmanr
from sklearn.metrics import jaccard_score

from misc.helpers import _append_and_create, _get_df, _get_glob_list
from misc.plotting import set_matplotlib_size, set_seaborn_style

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()


done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

del done_workload_df["EXP_RANDOM_SEED"]
del done_workload_df["EXP_NUM_QUERIES"]

column_combinations = [
    "EXP_DATASET",
    # "EXP_STRATEGY",
    "EXP_RANDOM_SEED",
    "EXP_START_POINT",
    "EXP_NUM_QUERIES",
    "EXP_BATCH_SIZE",
    "EXP_LEARNER_MODEL",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
]


print(f"Original: {len(done_workload_df)}")

# each row contains fingerprint -> I want to reduce this whole thing to a correlation among these fingerprints
# I calculate for each fingerprint, how good the individual strategies are -> I save the ranking values
# in the end I get a pd.DataFrame(colums=["fingerprint", "strat_a", "strat_b", "strat_c", …])
# and in each strat_a, strat_b, strat_c column I have the single_metric result for this strategy
# then I calculate the correlation between the time series of strategy results
# claudio frage: macht es Sinn darüber zu entscheiden, welche hyperparameter combinations ich verwenden soll?
"""dense_workload_grouped = done_workload_df.groupby(
    # by=[ddd for ddd in column_combinations if ddd != param_to_evaluate]
    by=column_combinations
).apply(lambda r: list(zip(r["EXP_STRATEGY"], r["EXP_UNIQUE_ID"])))

print(f"Calculating correlations for: {len(dense_workload_grouped)}")
"""


def _do_stuff(file_name: Path, config: Config, done_workload_df: pd.DataFrame):
    print(file_name)

    strategy_name = file_name.parent.parent.name

    metric_df = _get_df(file_name, config)

    if metric_df is None:
        return

    metric_df = pd.merge(metric_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left")

    return metric_df


glob_list = _get_glob_list(config, limit=f"**/full_auc_accuracy")

# metric_dfs = Parallel(n_jobs=1, verbose=10)(
metric_dfs = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name, config, done_workload_df) for file_name in glob_list
)

df = pd.concat(metric_dfs)
del df["EXP_UNIQUE_ID"]

non_al_cycle_keys = df.columns.difference(["0", "EXP_STRATEGY"])

# replace non_al_cycle_keys by single string fingerprint as key
df["fingerprint"] = df[non_al_cycle_keys].apply(
    lambda row: "_".join(row.values.astype(str)), axis=1
)

for non_al_cycle_key in non_al_cycle_keys:
    del df[non_al_cycle_key]

print(df)
df = df.pivot(index="fingerprint", columns="EXP_STRATEGY", values="0").reset_index()
df.columns.name = None
df = df.T

df.columns = df.loc["fingerprint"].values
df.drop(index="fingerprint", axis=0, inplace=True)

df = df.corr()
print(df)
print(non_al_cycle_keys)

result_folder = Path(config.OUTPUT_PATH / f"plots/")
result_folder.mkdir(parents=True, exist_ok=True)


df.to_parquet(result_folder / "fingerprint_correlations_based_on_single_metric.parquet")
# summed_up_corr_values.loc[:, "Total"] = summed_up_corr_values.mean(axis=1)
# summed_up_corr_values.sort_values(by=["Total"], inplace=True)

set_seaborn_style(font_size=8)
fig = plt.figure(figsize=set_matplotlib_size())
sns.heatmap(df, annot=True)

plt.savefig(
    result_folder / "fingerprint_correlations_based_on_single_metric.jpg",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
