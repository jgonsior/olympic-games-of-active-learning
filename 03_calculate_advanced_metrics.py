import importlib
from pathlib import Path

import pandas as pd
from misc.config import Config


config = Config()

print("computung the following metrics: " + ",".join(config.METRICS))


for computed_metric in config.METRICS:
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()
