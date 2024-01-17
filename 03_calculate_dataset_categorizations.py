import itertools
import multiprocessing
import sys
from typing import Tuple

from joblib import Parallel, delayed, parallel_backend


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
from resources.data_types import (
    SAMPLES_CATEGORIZER,
    samples_categorizer_to_classes_mapping,
)
from metrics.computed.base_samples_categorizer import *

pandarallel.initialize(progress_bar=True)

config = Config()

if config.SAMPLES_CATEGORIZER == ["_ALL"]:
    config.SAMPLES_CATEGORIZER = [sc.name for sc in SAMPLES_CATEGORIZER]

print(
    "computung the following samples categories: "
    + ",".join(config.SAMPLES_CATEGORIZER)
)


def _run_samples_categorizer(sc: str, ds: DATASET):
    print("#" * 100)
    print("computed_metric: " + str(sc) + str(ds))
    samples_categorizer_class = samples_categorizer_to_classes_mapping[
        SAMPLES_CATEGORIZER[sc]
    ](config)

    samples_categorizer_class.categorize_samples(ds)


dataset_categorizer_combinations: List[Tuple[DATASET, SAMPLES_CATEGORIZER]] = list(
    itertools.product(config.SAMPLES_CATEGORIZER, config.EXP_GRID_DATASET)
)


# with parallel_backend("threading", n_jobs=multiprocessing.cpu_count()):
#Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
Parallel(n_jobs=1, verbose=10)(
    delayed(_run_samples_categorizer)(sc, ds)
    for (sc, ds) in dataset_categorizer_combinations
)
