import multiprocessing
import sys
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from misc.plotting import set_matplotlib_size, set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

"""
for rank_or_percentage in ["dataset_normalized_percentages", "rank", "percentages"]:
    for grid_type in ["sparse", "dense"]:
        for interpolation in [
            "remove",
            "zero",
            "average_of_same_strategy",
        ]:
            if grid_type == "dense" and interpolation != "zero":
                continue

            if (
                rank_or_percentage == "dataset_normalized_percentages"
                and interpolation != "zero"
            ):
                continue

            for auc_prefix in [
                "full_auc_",
                # "first_5_",
                # "last_5_",
                # "ramp_up_auc_",
                # "plateau_auc_",
                # "final_value_",
            ]:
"""

ranking_path = Path(
    config.OUTPUT_PATH / f"plots/leaderboard_invariances/leaderboard_types.csv"
)

ranking_df = pd.read_csv(ranking_path, index_col=0).T
print(ranking_df)


for corr_method in ["spearman", "kendall"]:
    corr_data = ranking_df.corr(method=corr_method)

    destination_path = Path(
        config.OUTPUT_PATH
        / f"plots/leaderboard_invariances/leaderboard_types_{corr_method}"
    )

    print(str(destination_path) + f".jpg")
    set_seaborn_style(font_size=8)
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = 1.0
    # plt.figure(figsize=set_matplotlib_size(fraction=10))

    # calculate fraction based on length of keys
    plt.figure(figsize=set_matplotlib_size(fraction=len(corr_data.columns) / 6))

    ax = sns.heatmap(corr_data, annot=True, fmt=".2%")

    ax.set_title(f": {ranking_path}")

    corr_data.to_parquet(str(destination_path) + f".parquet")

    plt.savefig(
        str(destination_path) + f".jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
