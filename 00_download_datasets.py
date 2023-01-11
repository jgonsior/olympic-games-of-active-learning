import multiprocessing
from pathlib import Path
from datasets.kaggle import Kaggle
from datasets.localImporter import Local_Importer
from misc.config import Config
import glob
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import pairwise_distances

config = Config()

kaggle = Kaggle(config)
kaggle.load_datasets()

local_importer = Local_Importer(config)
local_importer.load_datasets()

# compute distances
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    dataset_csv_path = dataset_csv.replace(".csv", config.DATASETS_DISTANCES_APPENDIX)

    if Path(dataset_csv_path).exists():
        continue
    print("Computing distances for ", dataset_csv)

    df = pd.read_csv(dataset_csv, engine="pyarrow")
    X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
    distances = pairwise_distances(
        X, X, metric="cosine", n_jobs=multiprocessing.cpu_count()
    )

    dist_df = pd.DataFrame(distances)
    dist_df.to_csv(dataset_csv_path, index=False)


data = []
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    df = pd.read_csv(dataset_csv, engine="pyarrow")
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
