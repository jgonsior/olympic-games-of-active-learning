import csv
import itertools
from pathlib import Path
import pickle
import shutil
from timeit import timeit
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

amount = 10


def create_single_file():
    tmp_path = Path("tmp2/")
    tmp_path.mkdir(exist_ok=True)

    for iteration in range(0, amount):
        confusion_matrices = []
        selected_indices = []

        for _ in range(0, 50):
            selected_indices.append(np.random.randint(0, 2000, 5).tolist())
            y_true = np.random.randint(0, 10, 1000)
            y_pred = np.random.randint(0, 10, 1000)

            current_confusion_matrix = classification_report(
                y_true, y_pred, output_dict=True
            )

            confusion_matrices.append(current_confusion_matrix)

        # save metric results
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices

        columns = output_df.columns.to_list()
        index = output_df.index.to_list()
        flattened_exp_results_dict = {}
        for ix, col in itertools.product(index, columns):
            flattened_exp_results_dict[f"{ix}_{col}"] = output_df.iloc[ix][col]

        out_file = tmp_path / "results.csv"

        with open(out_file, "a") as f:
            w = csv.DictWriter(f, fieldnames=flattened_exp_results_dict.keys())

            if out_file.stat().st_size == 0:
                print("write headers first")
                w.writeheader()
            w.writerow(flattened_exp_results_dict)

    # read in csv and extract "overall accuracy"
    single_file = pd.read_csv(tmp_path / "results.csv", index_col=None)
    accs = len(single_file["49_accuracy"])
    print(accs)
    shutil.rmtree(tmp_path)


def create_files():
    tmp_path = Path("tmp2/")
    tmp_path.mkdir(exist_ok=True)

    for iteration in range(0, amount):
        confusion_matrices = []
        selected_indices = []

        for _ in range(0, 50):
            selected_indices.append(np.random.randint(0, 2000, 5).tolist())
            y_true = np.random.randint(0, 10, 1000)
            y_pred = np.random.randint(0, 10, 1000)

            current_confusion_matrix = classification_report(
                y_true, y_pred, output_dict=True
            )

            confusion_matrices.append(current_confusion_matrix)

        # save metric results
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices
        output_path = Path(tmp_path / f"{iteration}.feather")
        output_df.to_feather(output_path)

    accs = []
    # read all feather files
    for feather_file in tmp_path.glob("*.feather"):
        output_df = pd.read_feather(feather_file)
        accs.append(output_df.iloc[49]["accuracy"])
    accs = len(accs)
    print(accs)
    shutil.rmtree(tmp_path)


def create_pickles():
    tmp_path = Path("tmp2/")
    tmp_path.mkdir(exist_ok=True)

    for iteration in range(0, amount):
        confusion_matrices = []
        selected_indices = []

        for _ in range(0, 50):
            selected_indices.append(np.random.randint(0, 2000, 5).tolist())
            y_true = np.random.randint(0, 10, 1000)
            y_pred = np.random.randint(0, 10, 1000)

            current_confusion_matrix = classification_report(
                y_true, y_pred, output_dict=True
            )

            confusion_matrices.append(current_confusion_matrix)

        # save metric results
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices
        output_path = Path(tmp_path / f"{iteration}.feather")
        with open(output_path, "wb") as f:
            pickle.dump(output_df, f)
    accs = []
    # read all feather files
    for feather_file in tmp_path.glob("*.feather"):
        with open(feather_file, "rb") as f:
            output_df = pickle.load(f)
        accs.append(output_df.iloc[49]["accuracy"])
    accs = len(accs)
    print(accs)
    shutil.rmtree(tmp_path)


def create_many_csvs():
    tmp_path = Path("tmp2/")
    tmp_path.mkdir(exist_ok=True)

    for iteration in range(0, amount):
        confusion_matrices = []
        selected_indices = []

        for _ in range(0, 50):
            selected_indices.append(np.random.randint(0, 2000, 5).tolist())
            y_true = np.random.randint(0, 10, 1000)
            y_pred = np.random.randint(0, 10, 1000)

            current_confusion_matrix = classification_report(
                y_true, y_pred, output_dict=True
            )

            confusion_matrices.append(current_confusion_matrix)

        # save metric results
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices
        output_path = Path(tmp_path / f"{iteration}.feather")
        output_df.to_csv(output_path, index=None)

    accs = []
    # read all feather files
    for feather_file in tmp_path.glob("*.feather"):
        output_df = pd.read_csv(feather_file)
        accs.append(output_df.iloc[49]["accuracy"])
    accs = len(accs)
    print(accs)
    exit(-1)
    shutil.rmtree(tmp_path)


def get_all_accs():
    pass


# create_single_file()
# create_files()
print(timeit(create_many_csvs, number=10))
print(timeit(create_pickles, number=10))
print(timeit(create_files, number=10))
print(timeit(create_single_file, number=10))

# feather is egal, aber many files ist besser weil kein aufwändiges post processing
