import multiprocessing
import random
from typing import Any, List
import pandas as pd
from datasets import DATASET, load_dataset, split_dataset
from misc.config import Config
from misc.logging import log_it
from ressources.data_types import AL_STRATEGY, al_strategy_to_python_classes_mapping
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

config = Config()

# TODO: in order to run >1.000.000 jobs on taurus -> specify a random seeds range for this file to work through!

log_it(f"Executing Job # {config.WORKER_INDEX} of workload {config.WORKLOAD_FILE_PATH}")

random_seed_df = pd.read_csv(
    config.WORKLOAD_FILE_PATH,
    header=0,
    index_col=0,
    nrows=config.WORKER_INDEX + 1,
)
worker_dataset_id, worker_strategy_id, worker_random_seed = random_seed_df.loc[config.WORKER_INDEX]  # type: ignore

np.random.seed(worker_random_seed)
random.seed(worker_random_seed)


dataset = DATASET(worker_dataset_id)
strategy = AL_STRATEGY(worker_strategy_id)

log_it(
    f"Job Parameters are dataset_id {worker_dataset_id}-{dataset.name}, strategy_id {worker_strategy_id}-{strategy.name} and random_seed {worker_random_seed}"
)


# load dataset
df = load_dataset(dataset, config)
X, Y, train_idx, test_idx, label_idx, unlabel_idx = split_dataset(df, config)

# TODO change depending on config parameter
model = RandomForestClassifier(n_jobs=multiprocessing.cpu_count())

# initially train model on initally labeled data
model.fit(X=X[label_idx, :], y=Y[label_idx])  # type: ignore

# select the AL strategy to use
al_strategy = al_strategy_to_python_classes_mapping[strategy][0](
    X=X, y=Y, **al_strategy_to_python_classes_mapping[strategy][1]
)

# either we stop until all samples are labeled, or earlier
if config.EXP_NUM_QUERIES == 0:
    total_amount_of_iterations = int(len(unlabel_idx) / config.EXP_BATCH_SIZE) + 1
else:
    total_amount_of_iterations = config.EXP_NUM_QUERIES

# the metrics we want to analyze later on
accs: List[float] = []
f1_scores: List[float] = []
select_indices: List[np.ndarray] = []

# efficient list difference
def _list_difference(long_list: List[Any], short_list: List[Any]) -> List[Any]:
    short_set = set(short_list)
    return [i for i in long_list if not i in short_set]


log_it(f"Running for a total of {total_amount_of_iterations} iterations")

for iteration in range(0, total_amount_of_iterations):
    log_it(f"#{iteration}")

    # select some samples by indice to label
    select_ind = al_strategy.select(
        label_idx, unlabel_idx, model=model, batch_size=config.EXP_BATCH_SIZE
    )
    label_idx = label_idx + select_ind.tolist()
    unlabel_idx = _list_difference(unlabel_idx, select_ind.tolist())

    # save indices for later
    select_indices.append(select_ind.tolist())

    # update our learner model
    model.fit(X=X[label_idx, :], y=Y[label_idx])

    # prediction on test set for metrics
    pred = model.predict(X[test_idx, :])

    # calculate some metrics
    accuracy = accuracy_score(
        y_true=Y[test_idx],
        y_pred=pred,
    )
    accs.append(accuracy)

    f1 = f1_score(y_true=Y[test_idx], y_pred=pred, average="macro")
    f1_scores.append(f1)

    # TODO calculate more metrics
    # TODO calculate here only TP FP TN FN


# TODO store metrics
print(accs)
print(f1_scores)
print(select_indices)
