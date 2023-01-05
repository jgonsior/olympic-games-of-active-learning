import multiprocessing
from pathlib import Path
from datasets.kaggle import Kaggle
from datasets.localImporter import Local_Importer
from misc.config import Config
import glob
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import pairwise_distances
import numpy as np

config = Config()

kaggle = Kaggle(config)
kaggle.load_datasets()

local_importer = Local_Importer(config)
local_importer.load_datasets()

# compute distances
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    dataset_csv_path = dataset_csv.replace(".csv", "_distances.csv")

    if Path(dataset_csv_path).exists():
        continue

    df = pd.read_csv(dataset_csv)
    X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
    distances = pairwise_distances(
        X, X, metric="cosine", n_jobs=multiprocessing.cpu_count()
    )
    np.savez_compressed(dataset_csv_path, distances)


data = []
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    df = pd.read_csv(dataset_csv)
    data.append(
        [
            dataset_csv.replace(str(config.DATASETS_PATH), "")
            .replace("/", "")
            .replace(".csv", ""),
            f"({len(df.columns)}, {len(df)}, {len(df.LABEL_TARGET.unique())})",
        ]
    )
data_df = pd.DataFrame(data)
data_df.set_index(0, inplace=True)
data_df.sort_index(axis=0, inplace=True)
print(tabulate(data_df, tablefmt="fancy_grid"))
