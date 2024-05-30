import collections
from genericpath import isdir
import glob
import math
import multiprocessing
from operator import add
import subprocess
import sys
from matplotlib import pyplot as plt, rcParams
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size, set_seaborn_style
from resources.data_types import AL_STRATEGY
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()


plot_path = config.OUTPUT_PATH / "plots"

# print(plot_path)

for plot_folder in glob.glob(str(plot_path.resolve()) + "/**", recursive=True):
    plot_folder = Path(plot_folder)

    if plot_folder.is_file():
        continue

    plot_type_title = (
        str(plot_folder.resolve())
        .removeprefix(str(plot_path.resolve()))
        .removeprefix("/")
    )

    dfs = {}

    for parquet_file in glob.glob(str(plot_folder.resolve()) + "/*.parquet"):
        metric_title = str(Path(parquet_file).name).removesuffix(".parquet")
        df = pd.read_parquet(parquet_file)
        dfs[metric_title] = df

    if len(dfs) <= 1:
        continue

    dfs = collections.OrderedDict(sorted(dfs.items()))

    set_seaborn_style(font_size=5)

    ncols = 6
    nrows = math.ceil(len(dfs) / ncols)

    px = 1 / plt.rcParams["figure.dpi"]
    height = len(df) * nrows * 40 * px
    width = len(df) * ncols * 40 * px

    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        sharex="all",
        sharey="all",
        # figsize=(math.ceil(len(dfs) / nrows) * 20, nrows * 5),  # (breite, höhe)
        # figsize=(3840 * px, 1600 * px),
        figsize=(
            width,  # math.ceil(len(dfs) / nrows) * additional_sizing_factor * 8,
            height,  # nrows * additional_sizing_factor,
        ),
    )

    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    for i, (title, df) in enumerate(dfs.items()):
        axis = axs.flat[i]

        print(df)
        sns.heatmap(
            data=df,
            annot=True,
            fmt=".2f",
            ax=axis,
            vmin=0,
            vmax=1,
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            square=True,
        )
        axis.set_title(title, fontsize=plt.rcParams["figure.titlesize"])

    fig.suptitle(plot_type_title)

    plt.savefig(
        f"{config.OUTPUT_PATH}/plots/{plot_type_title.replace('/', '--')}_merged.pdf",
        # dpi=300,
    )
    print(plot_type_title)
