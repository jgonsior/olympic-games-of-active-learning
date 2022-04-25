import pandas as pd
from datasets.datasets import load_dataset
from misc.config import Config
from misc.logging import log_it

config = Config()

log_it(
    "Executing Job # {} of workload {}".format(
        config.WORKER_INDEX, config.WORKLOAD_FILE_PATH
    )
)

random_seed_df = pd.read_csv(
    config.WORKLOAD_FILE_PATH,
    header=0,
    index_col=0,
    nrows=config.WORKER_INDEX + 1,
)
worker_dataset_id, worker_strategy_id, worker_random_seed = random_seed_df.loc[config.WORKER_INDEX]  # type: ignore

log_it(
    "Job Parameters are dataset_id {}, strategy_id {} and random_seed {}".format(
        worker_dataset_id, worker_strategy_id, worker_random_seed
    )
)

# load dataset
load_dataset(worker_dataset_id, config)

# start AL experiment

# take a look at the mapping from id to dataset + id to strategy_id
# then run the experiment!
