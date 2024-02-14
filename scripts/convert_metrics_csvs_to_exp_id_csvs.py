import csv
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
config = Config()

parsed_metric_csv_file_path = Path(config.OUTPUT_PATH / "07_parsed_metric_csvs.csv")


if not parsed_metric_csv_file_path.exists():
    with open(parsed_metric_csv_file_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=["metric_file"])
        w.writeheader()

glob_list = [
    f
    for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True)
    if not f.endswith("_workload.csv.xz")
    and not f.endswith("_workloads.csv.xz")
    and not "/metrics/" in f
]

parsed_metric_csv_file = pd.read_csv(parsed_metric_csv_file_path)
parsed_metric_csvs = set(parsed_metric_csv_file["metric_file"].to_list())


print(len(glob_list))
glob_list = [ggg for ggg in glob_list if ggg not in parsed_metric_csvs]
print(len(glob_list))

# done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)

if not config.EXP_ID_METRIC_CSV_FOLDER_PATH.exists():
    config.EXP_ID_METRIC_CSV_FOLDER_PATH.mkdir()


def _do_stuff(file_name, config):
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_csv(metric_file)

    metric_name = str(metric_file.name).removesuffix(".csv.xz")

    for ix, row in df.iterrows():
        resulting_metric_file = (
            config.EXP_ID_METRIC_CSV_FOLDER_PATH / f"{int(row.EXP_UNIQUE_ID)}.csv.xz"
        )

        row_dict = {"metric": metric_name, **row.to_dict()}
        del row_dict["EXP_UNIQUE_ID"]

        if not resulting_metric_file.exists():
            with open(resulting_metric_file, "w") as f:
                w = csv.DictWriter(f, fieldnames=row_dict.keys())
                w.writeheader()

        with open(resulting_metric_file, "a") as f:
            w = csv.DictWriter(f, fieldnames=row_dict.keys())
            w.writerow(row_dict)

        # append new metric to existent file
        # print(resulting_metric_file)

    df.to_csv(file_name, index=False, compression="infer")

    with open(parsed_metric_csv_file_path, "a") as f:
        w = csv.DictWriter(f, fieldnames=["metric_file"])
        w.writerow({"metric_file": metric_file})


Parallel(n_jobs=1, verbose=10)(
    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    delayed(_do_stuff)(file_name, config)
    for file_name in glob_list
)
