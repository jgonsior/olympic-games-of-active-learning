import csv
from pathlib import Path
import sys
import glob

from joblib import Parallel, delayed
import pandas as pd

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

done_workload_df = pd.read_csv(config.OVERALL_DONE_WORKLOAD_PATH)
print(done_workload_df.keys())

ramp_plateau_results_file = config.DATASET_DEPENDENT_RANDOM_RAMP_PLATEAU_THRESHOLD_PATH


if not ramp_plateau_results_file.exists():
    with open(ramp_plateau_results_file, "a") as f:
        w = csv.DictWriter(f, fieldnames=["metric_file"])
        w.writeheader()


# read in accuracy files from random strategy, average them/smooth them
# check all buckets of 5 AL cycles for stationary
# when change from "stationary" to "non-stationary" --> we have a slope!

# EXP_DATASET
# EXP_BATCH_SIZE
# EXP_LEARNER_MODEL
# EXP_TRAIN_TEST_BUCKET_SIZE


def _do_stuff(file_name, config):
    metric_file = Path(file_name)
    print(metric_file)
    df = pd.read_csv(metric_file)
    print(df)

    # merge those rows together which belong together based on

    # EXP_DATASET
    # EXP_BATCH_SIZE
    # EXP_LEARNER_MODEL
    # EXP_TRAIN_TEST_BUCKET_SIZE

    exit(-1)


for EXP_DATASET in config.EXP_GRID_DATASET:
    if EXP_DATASET.name in ["Iris", "wine_origin"]:
        continue
    glob_list = [
        f
        for f in glob.glob(
            str(config.OUTPUT_PATH)
            + f"/ALIPY_RANDOM/{EXP_DATASET.name}/accuracy.csv.xz",
            recursive=True,
        )
    ]

    Parallel(n_jobs=1, verbose=10)(
        # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
        delayed(_do_stuff)(file_name, config)
        for file_name in glob_list
    )
