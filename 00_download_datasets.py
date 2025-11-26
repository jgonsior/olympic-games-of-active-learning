import glob
import multiprocessing
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import pairwise_distances
from tabulate import tabulate

from misc.config import Config

config = Config()

# openml_loader = OpenML_Loader(config)
# openml_loader.load_datasets()
# kaggle_loader = Kaggle_Loader(config)
# kaggle_loader.load_datasets()

# local_loader = Local_Loader(config)
# local_loader.load_datasets()

# compute distances
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    continue
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    dataset_csv_path = dataset_csv.replace(".csv", config.DATASETS_DISTANCES_APPENDIX)

    if Path(dataset_csv_path).exists():
        continue
    print("Computing distances for ", dataset_csv)

    df = pd.read_csv(dataset_csv)
    X = df.loc[:, df.columns != "LABEL_TARGET"].to_numpy()  # type: ignore
    distances = pairwise_distances(
        X, X, metric="cosine", n_jobs=multiprocessing.cpu_count()
    )

    dist_df = pd.DataFrame(distances)
    dist_df.to_csv(dataset_csv_path, index=False)


data = []

kaggle_parameter_dict: Dict[str, Any] = yaml.safe_load(
    config.KAGGLE_DATASETS_YAML_CONFIG_PATH.read_text()
)

openml_parameter_dict: Dict[str, Any] = yaml.safe_load(
    config.OPENML_DATASETS_YAML_CONFIG_PATH.read_text()
)

count_kaggle = 0
count_openml = 0
smallest_dataset = np.inf
biggest_dataset = 0
smallest_features = np.inf
biggest_features = 0
smallest_classes = np.inf
largest_classes = 0
count_binary = 0

kaggle_names = ""
openml_names = ""

for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX in dataset_csv:
        continue
    dataset_name = (
        dataset_csv.replace(str(config.DATASETS_PATH), "")
        .replace("/", "")
        .replace(".csv", "")
    )

    print(dataset_name)
    source = ""
    source_url = ""
    if dataset_name in kaggle_parameter_dict.keys():
        source += " kaggle repository:"
        source_url = kaggle_parameter_dict[dataset_name]["kaggle_name"]
        print("kaggle")
        kaggle_names += f"{source_url}, "
        count_kaggle += 1
    elif dataset_name in openml_parameter_dict.keys():
        source += " OpenML id:"
        source_url = openml_parameter_dict[dataset_name]["data_id"]
        openml_names += f"{dataset_name} ({source_url}), "
        print("openml")
        count_openml += 1
    else:
        source += " unknown"
        print("unknown")

    df = pd.read_csv(dataset_csv)
    if len(df) < smallest_dataset:
        smallest_dataset = len(df)
    if len(df) > biggest_dataset:
        biggest_dataset = len(df)

    if len(df.columns) == 3:
        count_binary += 1

    if len(df.columns) < smallest_features:
        smallest_features = len(df.columns) - 1
    if len(df.columns) > biggest_features:
        biggest_features = len(df.columns) - 1

    if len(df.LABEL_TARGET.unique()) < smallest_classes:
        smallest_classes = len(df.LABEL_TARGET.unique())
    if len(df.LABEL_TARGET.unique()) > largest_classes:
        largest_classes = len(df.LABEL_TARGET.unique())

    data.append(
        [
            dataset_name,
            f"{len(df):,}",
            f"{len(df.columns):,}",
            len(df.LABEL_TARGET.unique()),
            source,
            source_url,
            # f"({len(df.columns)}, {len(df)}, {len(df.LABEL_TARGET.unique())})",
        ]
    )
data_df = pd.DataFrame(data)
print(data_df)
data_df.set_index(0, inplace=True)
data_df.sort_index(axis=0, inplace=True)
print(tabulate(data_df, tablefmt="fancy_grid"))

with open("datasets.tex", "w") as f:
    f.write(tabulate(data_df, tablefmt="latex_booktabs"))

print(smallest_dataset)
print(biggest_dataset)
print(smallest_features)
print(biggest_features)
print(count_binary)
print(smallest_classes)
print(largest_classes)


print(kaggle_names)
print(openml_names)
