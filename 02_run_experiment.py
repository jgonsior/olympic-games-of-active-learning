import multiprocessing
import random
from typing import Any, List, cast
import pandas as pd
from datasets import DATASET, load_dataset, split_dataset
from misc.config import Config
from misc.logging import log_it
from ressources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
    al_strategy_to_python_classes_mapping,
    learner_models_to_classes_mapping,
)
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

config = Config()

# TODO: in order to run >1.000.000 jobs on taurus -> specify a random seeds range for this file to work through!

config.load_workload()

log_it(f"Executing Job # {config.WORKER_INDEX} of workload {config.WORKLOAD_FILE_PATH}")
np.random.seed(int(cast(int, config.EXP_RANDOM_SEED)))
random.seed(int(cast(int, config.EXP_RANDOM_SEED)))


dataset = DATASET(cast(DATASET, config.EXP_DATASET))

# load dataset
df = load_dataset(dataset, config)
X, Y, train_idx, test_idx, label_idx, unlabel_idx = split_dataset(df, config)

# load ml model
model_instantiation_tuple = learner_models_to_classes_mapping[
    cast(LEARNER_MODEL, config.EXP_LEARNER_MODEL)
]
model = model_instantiation_tuple[0](**model_instantiation_tuple[1])

# initially train model on initally labeled data
model.fit(X=X[label_idx, :], y=Y[label_idx])  # type: ignore

# select the AL strategy to use
al_strategy = AL_STRATEGY(cast(AL_STRATEGY, config.EXP_STRATEGY))
al_strategy = al_strategy_to_python_classes_mapping[al_strategy][0](
    X=X, y=Y, **al_strategy_to_python_classes_mapping[al_strategy][1]
)

# either we stop until all samples are labeled, or earlier
if cast(int, config.EXP_NUM_QUERIES) == 0:
    total_amount_of_iterations = (
        int(len(unlabel_idx) / cast(int, config.EXP_BATCH_SIZE)) + 1
    )
else:
    total_amount_of_iterations = cast(int, config.EXP_NUM_QUERIES)

# the metrics we want to analyze later on
confusion_matrices: List[np.ndarray] = []
selected_indices: List[np.ndarray] = []

# efficient list difference
def _list_difference(long_list: List[Any], short_list: List[Any]) -> List[Any]:
    short_set = set(short_list)
    return [i for i in long_list if not i in short_set]


log_it(f"Running for a total of {total_amount_of_iterations} iterations")

for iteration in range(0, total_amount_of_iterations):
    log_it(f"#{iteration}")

    # select some samples by indice to label
    select_ind = al_strategy.select(
        label_idx, unlabel_idx, model=model, batch_size=cast(int, config.EXP_BATCH_SIZE)
    )
    label_idx = label_idx + select_ind.tolist()
    unlabel_idx = _list_difference(unlabel_idx, select_ind.tolist())

    # save indices for later
    selected_indices.append(select_ind.tolist())

    # update our learner model
    model.fit(X=X[label_idx, :], y=Y[label_idx])

    # prediction on test set for metrics
    pred = model.predict(X[test_idx, :])

    current_confusion_matrix = classification_report(
        y_true=Y[test_idx], y_pred=pred, output_dict=True, zero_division=0
    )
    confusion_matrices.append(current_confusion_matrix)


# TODO
output_df = pd.DataFrame(
    {"selected_indices": selected_indices, "confusion_matrix": confusion_matrices}
)
print(output_df)
