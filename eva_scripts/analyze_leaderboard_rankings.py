import multiprocessing
import sys
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from misc.plotting import _rename_strategy, set_matplotlib_size, set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()
from pandarallel import pandarallel

pandarallel.initialize(
    nb_workers=multiprocessing.cpu_count(), progress_bar=True, use_memory_fs=False
)

ranking_path = Path(
    config.OUTPUT_PATH / f"plots/leaderboard_invariances/leaderboard_types.csv"
)
ranking_df = pd.read_csv(ranking_path, index_col=0)
ranking_df.rename(columns=_rename_strategy, inplace=True)
ranking_df = ranking_df.T


# for corr_method in ["spearman", "kendall"]:
for corr_method in ["kendall"]:
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

    ax = sns.heatmap(corr_data, annot=True, fmt=".2%", square=True)

    ax.set_title(f": {ranking_path}")

    corr_data.to_parquet(str(destination_path) + f".parquet")

    plt.savefig(
        str(destination_path) + f".jpg",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
