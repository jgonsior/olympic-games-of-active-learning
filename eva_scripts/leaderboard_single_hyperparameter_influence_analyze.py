import multiprocessing
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import scipy
from datasets import DATASET
from misc.plotting import _rename_strategy, set_matplotlib_size, set_seaborn_style
from resources.data_types import LEARNER_MODEL
import seaborn as sns
import ast
import matplotlib.ticker as ticker

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

combined_plot = False

if combined_plot:
    hyperparameters_to_evaluate = [
        # "adv_start_scenario",
        # "start_point_scenario",
        # "dataset_scenario",
        "standard_metric",
        # "EXP_STRATEGY",
        "EXP_LEARNER_MODEL",
        "EXP_BATCH_SIZE",
        # "EXP_DATASET",
        "EXP_TRAIN_TEST_BUCKET_SIZE",
        # "EXP_START_POINT",
        "auc_metric",
    ]

    rankings_df: pd.DataFrame = pd.DataFrame()
    for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
        ranking_path = Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
        )
        ranking_df = pd.read_csv(ranking_path, index_col=0)
        ranking_df.rename(columns=_rename_strategy, inplace=True)
        ranking_df = ranking_df.T

        keys = {
            kkk: kkk.removeprefix(f"{hyperparameter_to_evaluate}: ")
            for kkk in ranking_df.columns
        }

        ranking_df.rename(columns=keys, inplace=True)

        if hyperparameter_to_evaluate == "EXP_LEARNER_MODEL":
            keys = {kkk: LEARNER_MODEL(int(kkk)).name for kkk in ranking_df.columns}
            ranking_df.rename(columns=keys, inplace=True)
        elif hyperparameter_to_evaluate == "EXP_BATCH_SIZE":
            keys = {kkk: int(kkk) for kkk in ranking_df.columns}
            ranking_df.rename(columns=keys, inplace=True)
        elif hyperparameter_to_evaluate == "EXP_DATASET":
            keys = {kkk: DATASET(int(kkk)).name for kkk in ranking_df.columns}
            ranking_df.rename(columns=keys, inplace=True)

        if hyperparameter_to_evaluate in [
            "adv_min",
            "min_hyper",
            "min_hyper2",
            "adv_start_scenario",
            "start_point_scenario",
            "dataset_scenario",
        ]:
            custom_dict = {
                v: k
                for k, v in enumerate(
                    sorted(
                        ranking_df.columns,
                        key=lambda kkk: int(ast.literal_eval(kkk)[1]),
                    )
                )
            }
            ranking_df = ranking_df.sort_index(axis=0)
            ranking_df = ranking_df.sort_index(key=lambda x: x.map(custom_dict), axis=1)
        else:
            ranking_df = ranking_df.sort_index(axis=0)
            ranking_df = ranking_df.sort_index(axis=1)

        keys = {
            kkk: f"{hyperparameter_to_evaluate}: {kkk}" for kkk in ranking_df.columns
        }

        ranking_df.rename(columns=keys, inplace=True)

        if len(rankings_df) == 0:
            rankings_df = ranking_df.T
        else:
            rankings_df = pd.concat([rankings_df, ranking_df.T])
    rankings_df = rankings_df.T

    # convert into ranks
    def _calculate_ranks(row: pd.Series) -> pd.Series:
        ranks = scipy.stats.rankdata(row, method="max", nan_policy="omit")
        result = pd.Series(ranks, index=row.index)
        return result

    rankings_df = rankings_df.parallel_apply(_calculate_ranks, axis=0)

    # calculate kendall and speraman as last row
    # sort x-axis after last row, sort y-axis after gold standard

    rankings_df.rename(
        columns={"standard_metric: full_auc_weighted_f1-score": "gold standard"},
        inplace=True,
    )

    rankings_df.sort_values("gold standard", inplace=True)

    def _calculate_spearman(row: pd.Series) -> pd.Series:
        kendalltau = scipy.stats.kendalltau(row, rankings_df.loc["gold standard", :])
        # kendalltau = scipy.stats.spearmanr(row, rankings_df.loc["gold standard", :])

        res = np.nan
        if kendalltau.pvalue < 0.05:
            res = kendalltau.statistic
        return res

    rankings_df = rankings_df.T

    rankings_df["spearman"] = rankings_df.apply(_calculate_spearman, axis=1)
    rankings_df = rankings_df.T
    rankings_df.sort_values(by="spearman", axis=1, inplace=True)

    destination_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_single_hyperparameter_influence/all_together"
    )

    print(str(destination_path) + f".jpg")
    set_seaborn_style(font_size=8)
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = 1.0
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    plt.figure(
        figsize=set_matplotlib_size(
            fraction=len(rankings_df.columns) / 20, half_height=True
        ),
    )

    ax = sns.heatmap(
        rankings_df,
        annot=True,
        fmt="g",
        cmap=sns.color_palette("husl", len(rankings_df) - 1),
    )

    ax.set_title(f"{hyperparameter_to_evaluate}")

    # rankings_df.to_parquet(str(destination_path) + f".parquet")

    plt.savefig(
        str(destination_path) + f".jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

hyperparameters_to_evaluate = [
    "min_hyper2",
    ("min_hyper_reduction", "EXP_START_POINT"),
    ("min_hyper_reduction", "EXP_TRAIN_TEST_BUCKET_SIZE"),
    ("min_hyper_reduction", "EXP_BATCH_SIZE"),
    ("min_hyper_reduction", "EXP_LEARNER_MODEL"),
    "min_hyper233",
    "min_hyper",
    "adv_min",
    "adv_start_scenario",
    "dataset_scenario",
    "start_point_scenario",
    "standard_metric",
    # "EXP_STRATEGY",
    "EXP_START_POINT",
    "EXP_LEARNER_MODEL",
    "EXP_BATCH_SIZE",
    "EXP_DATASET",
    "EXP_TRAIN_TEST_BUCKET_SIZE",
    "auc_metric",
]

hyperparameter_to_evaluate_addendum = False
for hyperparameter_to_evaluate in hyperparameters_to_evaluate:
    if len(hyperparameter_to_evaluate) == 2:
        hyperparameter_to_evaluate_addendum = hyperparameter_to_evaluate[1]
        hyperparameter_to_evaluate = hyperparameter_to_evaluate[0]

        decomposed_path = Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_decomposed.csv"
        )

        if not decomposed_path.exists():
            df = pd.read_csv(
                Path(
                    config.OUTPUT_PATH
                    / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
                ),
                index_col=0,
            )
            df.reset_index(inplace=True)

            def _decompose_titles(row: pd.Series) -> pd.Series:
                name = ast.literal_eval(
                    str(row["index"]).removeprefix("min_hyper_reduction: ")
                )

                row["index"] = f"min_hyper_reduction: ({name[0]}, {name[1]})"
                row["parameter"] = name[2]
                return row

            df = df.parallel_apply(_decompose_titles, axis=1)
            df.set_index("index", inplace=True)
            df.to_csv(decomposed_path)

        ranking_df = pd.read_csv(decomposed_path, index_col=0)
        ranking_df = ranking_df.loc[
            ranking_df["parameter"] == hyperparameter_to_evaluate_addendum
        ]

        del ranking_df["parameter"]

    else:
        ranking_path = Path(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}.csv"
        )

        print(f"reading in {ranking_path}")
        ranking_df = pd.read_csv(ranking_path, index_col=0)
        print("finished reading")

    ranking_df.rename(columns=_rename_strategy, inplace=True)

    ranking_df = ranking_df.T

    keys = {
        kkk: kkk.removeprefix(f"{hyperparameter_to_evaluate}: ")
        for kkk in ranking_df.columns
    }
    ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate == "EXP_LEARNER_MODEL":
        keys = {kkk: LEARNER_MODEL(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_BATCH_SIZE":
        keys = {kkk: int(kkk) for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)
    elif hyperparameter_to_evaluate == "EXP_DATASET":
        keys = {kkk: DATASET(int(kkk)).name for kkk in ranking_df.columns}
        ranking_df.rename(columns=keys, inplace=True)

    if hyperparameter_to_evaluate in [
        "adv_min",
        "min_hyper",
        "min_hyper2",
        "min_hyper_reduction",
        "adv_start_scenario",
        "start_point_scenario",
        "dataset_scenario",
    ]:
        custom_dict = {
            v: k
            for k, v in enumerate(
                sorted(
                    ranking_df.columns, key=lambda kkk: int(ast.literal_eval(kkk)[1])
                )
            )
        }
        ranking_df = ranking_df.sort_index(axis=0)

        ranking_df = ranking_df.sort_index(key=lambda x: x.map(custom_dict), axis=1)

    else:
        ranking_df = ranking_df.sort_index(axis=0)
        ranking_df = ranking_df.sort_index(axis=1)

    if hyperparameter_to_evaluate == "min_hyper2":
        # rename_dict = {kkk: ast.literal_eval(kkk)[1] for kkk in ranking_df.columns}
        # ranking_df.rename(columns=rename_dict, inplace=True)
        # rename_dict = {kkk: kkk if kkk < 10000 else -1 for kkk in ranking_df.columns}
        # ranking_df.rename(columns=rename_dict, inplace=True)
        # del ranking_df[-1]

        # nr_buckets = 10000
        # min_value = ranking_df.columns[0]
        # max_value = ranking_df.columns[-1]

        # buckets = {
        #    vvv: f"({vvv}, {k})"
        #    for k, v in enumerate(
        #        np.array_split(range(min_value, max_value + 1), nr_buckets)
        #    )
        #    for vvv in v
        # }
        # ranking_df.rename(columns=buckets, inplace=True)
        # print(ranking_df)

        # buckets = {vvv: f"({vvv}, {vvv})" for vvv in ranking_df.columns}
        # ranking_df.rename(columns=buckets, inplace=True)

        ranking_df.drop(
            ranking_df.columns[len(ranking_df.columns) - 1], axis=1, inplace=True
        )

        # exit(-1)
        #  ranking_df.rename(
        #  columns={ranking_df.columns[-1]: f"(420000, {ranking_df.columns[-1]})"},
        #  inplace=True,
        #  )

    if hyperparameter_to_evaluate == "adv_min":
        buckets = {
            kkk: (ast.literal_eval(kkk)[0], ast.literal_eval(kkk)[2])
            for kkk in ranking_df.columns
        }
        ranking_df.rename(columns=buckets, inplace=True)

    if hyperparameter_to_evaluate == "adv_start_scenario":
        ranking_df.rename(columns={f"(0, 21)": "gold standard"}, inplace=True)
    else:
        # add gold standard
        gold_standard = pd.read_csv(
            config.OUTPUT_PATH
            / f"plots/leaderboard_single_hyperparameter_influence/standard_metric.csv",
            index_col=0,
        )
        gold_standard.rename(columns=_rename_strategy, inplace=True)

        ranking_df["gold standard"] = gold_standard.loc[
            "standard_metric: full_auc_weighted_f1-score"
        ]

    ranking_df.rename(
        columns={kkk: str(kkk) for kkk in ranking_df.columns}, inplace=True
    )

    for hypothesis in [
        # "pearson",
        "kendall",
        # "spearman",
        # "kendall_unc_better_than_repr",
        # "same strategies - same rank"
        # "mm_better_lc_then_ent",
        # "random_similar",
        # "optimal best",
        # "quire similar"
        # "same strategy but in different frameworks behave similar"
    ]:
        # check how "well" the hypothesis can be found in the rankings!

        if hyperparameter_to_evaluate in [
            "adv_min",
            "min_hyper",
            "min_hyper2",
            "adv_start_scenario",
            "start_point_scenario",
            "dataset_scenario",
            "min_hyper_reduction",
        ]:

            def _calculate_spearman(row: pd.Series) -> pd.Series:
                kendalltau = scipy.stats.kendalltau(
                    row, ranking_df.loc["gold standard", :]
                )
                # kendalltau = scipy.stats.spearmanr(row, rankings_df.loc["gold standard", :])

                # res = np.nan
                # if kendalltau.pvalue < 0.05:
                #    res = kendalltau.statistic
                res = kendalltau.statistic
                return res

            ranking_df = ranking_df.T

            ranking_df["spearman"] = ranking_df.parallel_apply(
                _calculate_spearman, axis=1
            )

            ranking_df = ranking_df.T
            ranking_df.sort_values(by="spearman", axis=1, inplace=True)
            ranking_df = ranking_df.T[["spearman"]]
            ranking_df = ranking_df.reset_index()

            gold_standard = ranking_df.loc[ranking_df["index"] == "gold standard"]
            ranking_df = ranking_df[ranking_df["index"] != "gold standard"]
            print(ranking_df)
            ranking_df["index"] = ranking_df["index"].parallel_apply(
                lambda kkk: ast.literal_eval(kkk)[1]
            )
            ranking_df.sort_values(by="index", inplace=True)
            #  ranking_df = pd.concat([ranking_df, gold_standard])
            corr_data = ranking_df
            """
            grouped_values = defaultdict(list)
            for ix, spr in ranking_df.loc["spearman"].items():
                if ix == "gold standard":
                    continue
                grouped_values[ast.literal_eval(ix)[1]].append(spr)


            for k, v in grouped_values.items():
                grouped_values[k] = [np.mean(v), np.std(v)]
            corr_data = pd.DataFrame(grouped_values, index=["mean", "std"])
            corr_data.sort_index(axis=0, inplace=True)
            corr_data.sort_index(axis=1, inplace=True)
            """
        else:
            corr_data = ranking_df.corr(method=hypothesis)

        print(ranking_df)

        if hyperparameter_to_evaluate == "min_hyper_reduction":
            destination_path = Path(
                config.OUTPUT_PATH
                / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_{hyperparameter_to_evaluate_addendum}_{hypothesis}"
            )
        else:
            destination_path = Path(
                config.OUTPUT_PATH
                / f"plots/leaderboard_single_hyperparameter_influence/{hyperparameter_to_evaluate}_{hypothesis}"
            )

        print(str(destination_path) + f".jpg")
        set_seaborn_style(font_size=8)
        mpl.rcParams["path.simplify"] = True
        mpl.rcParams["path.simplify_threshold"] = 1.0
        # plt.figure(figsize=set_matplotlib_size(fraction=10))

        # calculate fraction based on length of keys
        plt.figure(
            figsize=set_matplotlib_size(fraction=10)
        )  # fraction=len(corr_data.columns) / 6))

        if hyperparameter_to_evaluate in [
            "adv_min",
            "min_hyper",
            "min_hyper2",
            "adv_start_scenario",
            "start_point_scenario",
            "dataset_scenario",
            "min_hyper_reduction",
        ]:
            # calculate fraction based on length of keys
            plt.figure(figsize=set_matplotlib_size(fraction=1))
            ax = sns.lineplot(
                data=corr_data,
                x="index",
                y="spearman",
                errorbar=lambda x: (x.min(), x.max()),
                #  errorbar="sd",:w
                #  sizes=(0.1, 0.1),
                #  alpha=0.2,
                #  edgecolor="none",
                #  hue=0.3,
            )
            #  ax.xaxis.set_major_locator(ticker.LinearLocator(20))
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            ax.set(ylim=(0, 1))
            # ax = sns.violinplot(data=corr_data, x="index", y="spearman", hue="index")
        else:
            # calculate fraction based on length of keys
            plt.figure(figsize=set_matplotlib_size(fraction=len(corr_data.columns) / 6))

            ax = sns.heatmap(corr_data, annot=True, fmt=".2%", vmin=0, vmax=1)

        # ax.set_title(f"{hyperparameter_to_evaluate}")
        plt.legend([], [], frameon=False)

        if hyperparameter_to_evaluate in [
            "adv_min",
            "min_hyper",
            "min_hyper2",
            "adv_start_scenario",
            "start_point_scenario",
            "dataset_scenario",
            "min_hyper_reduction",
        ]:
            corr_data["index"] = corr_data["index"].parallel_apply(str)
        corr_data.to_parquet(str(destination_path) + f".parquet")

        plt.savefig(
            str(destination_path) + f".jpg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.savefig(
            str(destination_path) + f".pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.clf()
