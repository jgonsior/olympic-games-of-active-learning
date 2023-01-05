import importlib

from misc.config import Config


config = Config()

print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))


for computed_metric in config.COMPUTED_METRICS:
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()
