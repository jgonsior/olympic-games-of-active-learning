# über alle selected_indices.csvs drüber iterieren
# an done_workload_df dran joinen
# in große dataframe speichern wie das "grouped" ergebnis aussieht
# ich will ja vregleichen, alles wird festgehalten, und nur die strategien verändern sich, wie sieht es dann mit der correlation zwischen den selected indices aus
# am ende ergebnis in csv und plotten mit heatmap, jaccard

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


def _flatten(xss):
    return [x for xs in xss for x in xs]


def _do_stuff(exp_dataset, config):
    glob_list = _get_glob_list(config, limit=f"**/{exp_dataset.name}/selected_indices")

    if len(glob_list) == 0:
        return

    selected_indices_per_strategy = {}
    for file_name in glob_list:
        print(file_name)

        strategy_name = file_name.parent.parent.name

        selected_indices_df = _get_df(file_name, config)

        if selected_indices_df is None:
            return

        cols_with_indice_lists = selected_indices_df.columns.difference(
            ["EXP_UNIQUE_ID"]
        )

        selected_indices_df[cols_with_indice_lists] = (
            selected_indices_df[cols_with_indice_lists]
            .fillna("[]")
            .map(lambda x: ast.literal_eval(x))
        )

        selected_indices_df = pd.merge(
            selected_indices_df, done_workload_df, on=["EXP_UNIQUE_ID"], how="left"
        )

        del selected_indices_df["EXP_UNIQUE_ID"]
        del selected_indices_df["EXP_STRATEGY"]

        non_al_cycle_keys = selected_indices_df.columns.difference(
            cols_with_indice_lists
        )

        # replace non_al_cycle_keys by single string fingerprint as key
        selected_indices_df["fingerprint"] = selected_indices_df[
            non_al_cycle_keys
        ].apply(lambda row: "_".join(row.values.astype(str)), axis=1)

        selected_indices_df["selected_indices"] = selected_indices_df[
            cols_with_indice_lists
        ].apply(lambda row: _flatten(row.values.tolist()), axis=1)

        for non_al_cycle_key in non_al_cycle_keys:
            del selected_indices_df[non_al_cycle_key]
        for non_al_cycle_key in cols_with_indice_lists:
            del selected_indices_df[non_al_cycle_key]

        selected_indices_per_strategy[strategy_name] = selected_indices_df

    shared_fingerprints = []

    for selected_indices_df in selected_indices_per_strategy.values():
        shared_fingerprints.append(set(selected_indices_df["fingerprint"].to_list()))
    shared_fingerprints = set.intersection(*shared_fingerprints)
    table_data = []
    for strat_a, strat_b in itertools.combinations(
        selected_indices_per_strategy.keys(), 2
    ):
        jaccards = []
        spearmans = []

        for shared_fingerprint in shared_fingerprints:
            queried_a = (
                selected_indices_per_strategy[strat_a]
                .loc[
                    selected_indices_per_strategy[strat_a]["fingerprint"]
                    == shared_fingerprint
                ]["selected_indices"]
                .iloc[0]
            )
            queried_b = (
                selected_indices_per_strategy[strat_b]
                .loc[
                    selected_indices_per_strategy[strat_b]["fingerprint"]
                    == shared_fingerprint
                ]["selected_indices"]
                .iloc[0]
            )
            jaccard = len(np.intersect1d(queried_a, queried_b)) / len(
                np.union1d(queried_a, queried_b)
            )
            spearman = spearmanr(queried_a, queried_b)[0]
            jaccards.append(jaccard)
            spearmans.append(spearman)

        table_data.append(
            (
                strat_a,
                strat_b,
                jaccards,
                spearmans,
                # np.mean(jaccards),
                # np.std(jaccards),
            )
        )

        table_data.append(
            (
                strat_b,
                strat_a,
                jaccards,
                spearmans,
                # np.mean(jaccards),
                # np.std(jaccards),
            )
        )

    for strat in selected_indices_per_strategy.keys():
        table_data.append((strat, strat, [1], [1]))
    return table_data


# table_datas = Parallel(n_jobs=1, verbose=10)(
table_datas = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(exp_dataset, config) for exp_dataset in config.EXP_GRID_DATASET
)


df = None

for table_data in table_datas:
    single_df = pd.DataFrame(
        table_data, columns=["strat_a", "strat_b", "jaccards", "spearmans"]
    )

    if df is None:
        df = single_df
        continue

    df = pd.merge(single_df, df, on=["strat_a", "strat_b"], how="outer")
    df["jaccards_x"] = df["jaccards_x"].fillna("").apply(list)
    df["spearmans_x"] = df["spearmans_x"].fillna("").apply(list)

    df["jaccards"] = df[["jaccards_x", "jaccards_y"]].apply(
        lambda row: [*row["jaccards_x"], *row["jaccards_y"]], axis=1
    )
    df["spearmans"] = df[["spearmans_x", "spearmans_y"]].apply(
        lambda row: [*row["spearmans_x"], *row["spearmans_y"]], axis=1
    )
    del df["jaccards_x"]
    del df["jaccards_y"]
    del df["spearmans_x"]
    del df["spearmans_y"]

df = df.pivot(index="strat_a", columns="strat_b", values="spearmans")

print(df)

result_folder = Path(config.OUTPUT_PATH / f"plots/")
result_folder.mkdir(parents=True, exist_ok=True)

df.to_parquet(result_folder / "similar_strategies.parquet")
df = df.map(lambda r: np.mean(r))
# summed_up_corr_values.loc[:, "Total"] = summed_up_corr_values.mean(axis=1)
# summed_up_corr_values.sort_values(by=["Total"], inplace=True)

print(df)

set_seaborn_style(font_size=8)
fig = plt.figure(figsize=set_matplotlib_size())
sns.heatmap(df, annot=True)

plt.savefig(
    result_folder / "similar_strategies.jpg", dpi=300, bbox_inches="tight", pad_inches=0
)
