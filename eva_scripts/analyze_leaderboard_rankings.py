import csv
import multiprocessing
import subprocess
import sys
from typing import Dict
from matplotlib import pyplot as plt, transforms
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.style as mplstyle
from sklearn.preprocessing import RobustScaler
import matplotlib as mpl
import scipy
from datasets import DATASET
from misc.helpers import (
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_matplotlib_size, set_seaborn_style
from resources.data_types import AL_STRATEGY
import seaborn as sns
from pprint import pprint

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

print(ranking_df.corr(method="spearman"))
print(ranking_df.corr(method="kendall"))
