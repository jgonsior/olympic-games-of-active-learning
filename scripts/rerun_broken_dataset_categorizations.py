import ast
import csv
import multiprocessing
from pathlib import Path
import sys
import glob
import lzma

from joblib import Parallel, delayed
import pandas as pd

from resources.data_types import AL_STRATEGY, SAMPLES_CATEGORIZER

sys.dont_write_bytecode = True

from datasets import DATASET
from misc.config import Config
from pandarallel import pandarallel
import shutil

pandarallel.initialize(progress_bar=True)
config = Config()


broken_csvs = pd.read_csv(config.OUTPUT_PATH / "07_broken_csvs.csv")

open_dataset_categorization_path = Path(
    config.OUTPUT_PATH / "workloads/advanced_metrics/01_open.csv"
)


for _, broken_csv in broken_csvs.iterrows():
    broken_csv = Path(broken_csv["metric_file"])
    dataset_categorization = broken_csv.name.removesuffix(".csv.xz")

    dc_list = [sc.name for sc in SAMPLES_CATEGORIZER]
    if dataset_categorization in dc_list:
        dataset_categorization = "DATASET_CATEGORIZATION"

    strategy_enum_int = AL_STRATEGY[broken_csv.parent.parent.name]
    dataset_enum_int = DATASET[broken_csv.parent.name]

    with open(open_dataset_categorization_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=["0", "1", "2"])
        w.writerow(
            {"0": dataset_categorization, "1": strategy_enum_int, "2": dataset_enum_int}
        )
