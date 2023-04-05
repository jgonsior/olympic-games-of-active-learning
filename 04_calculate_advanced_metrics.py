import sys

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

for computed_metric in config.COMPUTED_METRICS:
    print()
    print()
    print()
    print("###" * 100)
    print(computed_metric)
    print("###" * 100)
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()
