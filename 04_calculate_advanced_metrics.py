import multiprocessing
import sys

from joblib import Parallel, delayed

from resources.data_types import COMPUTED_METRIC

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

config = Config()


if config.COMPUTED_METRICS == ["_ALL"]:
    # ensure that STANDARD_AUC is run last so that all other metrics are already precomputed
    config.COMPUTED_METRICS = [
        sc.name
        for sc in COMPUTED_METRIC
        if sc != COMPUTED_METRIC.STANDARD_AUC and sc != COMPUTED_METRIC.TIMELAG_METRIC
    ] + [COMPUTED_METRIC.TIMELAG_METRIC.name, COMPUTED_METRIC.STANDARD_AUC.name]


print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))


# -> hier erstmal eine liste mit den combinations von metriken etc. erstellen --> dann schön gepflegt aufrufen das ganze
print(config.COMPUTED_METRICS)


def _run_single_metric(m):
    m[0](*m[1:])


for computed_metric in config.COMPUTED_METRICS:
    # print(computed_metric)
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    metrics_to_compute = computed_metric_class.compute()

    # Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
    Parallel(
        n_jobs=8,
        backend="multiprocessing",
        verbose=10
        # n_jobs=multiprocessing.cpu_count(), backend="multiprocessing", verbose=10
    )(  # multiprocessing.cpu_count(), verbose=10)(
        delayed(_run_single_metric)(m) for (m) in metrics_to_compute
    )
