import itertools
import multiprocessing
import sys
from typing import Tuple

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel
from resources.data_types import (
    SAMPLES_CATEGORIZER,
    COMPUTED_METRIC,
    samples_categorizer_to_classes_mapping,
)
from metrics.computed.base_samples_categorizer import *

pandarallel.initialize(progress_bar=True)

config = Config()


print(
    "computung the following samples categories: "
    + ",".join(config.SAMPLES_CATEGORIZER)
)
if config.SAMPLES_CATEGORIZER == ["_ALL"]:
    config.SAMPLES_CATEGORIZER = [sc for sc in SAMPLES_CATEGORIZER]


def _run_samples_categorizer(sc: SAMPLES_CATEGORIZER, ds: DATASET):
    print("#" * 100)
    print("computed_metric: " + str(sc))
    samples_categorizer_class = samples_categorizer_to_classes_mapping[sc](config)

    samples_categorizer_class.categorize_samples(ds)


dataset_categorizer_combinations: List[Tuple[DATASET, SAMPLES_CATEGORIZER]] = list(
    itertools.product(config.SAMPLES_CATEGORIZER, config.EXP_GRID_DATASET)
)

with parallel_backend("loky", n_jobs=1):  # multiprocessing.cpu_count()):
    Parallel()(
        delayed(_run_samples_categorizer)(sc, ds)
        for (sc, ds) in dataset_categorizer_combinations
    )


exit(-1)
print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))
for computed_metric in config.COMPUTED_METRICS:
    print("#" * 100)
    print("computed_metric: " + str(computed_metric))
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()


"""def run_code(i):
    cli = f"python 02_run_experiment.py --EXP_TITLE local_SynDs --WORKER_INDEX {i}"
    print("#" * 100)
    print(i)
    print(cli)
    print("#" * 100)
    print("\n")
    os.system(cli)


with parallel_backend("loky", n_jobs=multiprocessing.cpu_count()):
    Parallel()(delayed(run_code)(i) for i in range(0, 1680))"""
