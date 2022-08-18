from datasets.kaggle import Kaggle
from misc.config import Config
import glob
import pandas as pd
from tabulate import tabulate

config = Config()

kaggle = Kaggle(config)
kaggle.download_all_datasets()

print(str(config.DATASETS_PATH) + "/*.csv")

data = []
for dataset_csv in list(glob.glob(str(config.DATASETS_PATH) + "/*.csv")):
    if "_train_test_split" in dataset_csv:
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
