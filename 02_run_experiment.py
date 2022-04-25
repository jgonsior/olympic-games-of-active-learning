import itertools
import pandas as pd
from misc.config import Config
from misc.logging import log_it
import os

config = Config()
print(config.WORKER_INDEX)
random_seed_df = pd.read_csv(
    config.WORKLOAD_FILE_PATH,
    header=0,
    index_col=0,
    nrows=config.WORKER_INDEX + 1,
)
print(random_seed_df)
worker_dataset_id, worker_strategy_id, worker_random_seed = random_seed_df.loc[config.WORKER_INDEX]  # type: ignore

print(worker_dataset_id)
print(worker_strategy_id)
print(worker_random_seed)


# take a look at the mapping from id to dataset + id to strategy_id
# then run the experiment!
