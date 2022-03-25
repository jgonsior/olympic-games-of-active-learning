import itertools
import pandas as pd
from misc.config import Config
from misc.logging import log_it
import os

# usage example: python 01_create_workload.py --EXP_DATASETS 1,2,3,4,5,6 --EXP_STRATEGIES 5,10 --EXP_RANDOM_SEEDS 100

config = Config()


# check results
if os.path.isfile(config.EXP_RESULTS_FILE):
    result_df = pd.read_csv(
        config.EXP_RESULTS_FILE,
        index_col=None,
        usecols=["dataset_id", "strategy_id", "dataset_random_seed"],
    )
else:
    result_df = pd.DataFrame(
        data=None, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
    )

missing_ids = []

for dataset_id, strategy_id, dataset_random_seed in itertools.product(
    config.EXP_DATASETS,
    config.EXP_STRATEGIES,
    config.EXP_RANDOM_SEEDS,
    repeat=1,
):
    if (
        len(
            result_df.loc[
                (result_df["dataset_id"] == dataset_id)
                & (result_df["strategy_id"] == strategy_id)
                & (result_df["dataset_random_seed"] == dataset_random_seed)
            ]
        )
        == 0
    ):
        missing_ids.append([dataset_id, strategy_id, dataset_random_seed])


random_seed_df = pd.DataFrame(
    data=missing_ids, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
)

random_seed_df.to_csv(config.OUTPUT_PATH + "/workload.csv", header=True)
config.save_to_file()
