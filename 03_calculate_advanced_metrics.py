import importlib
from pathlib import Path

import pandas as pd
from misc.config import Config


config = Config()

# easy: calculate f1-auc
print(config.METRICS)


def w_f1_auc(config: Config):
    for EXP_STRATEGY in config.EXP_GRID_STRATEGY:
        for EXP_DATASET in config.EXP_GRID_DATASET:
            # iterate over all experiments/datasets defined for this experiment
            METRIC_RESULTS_FILE = Path(
                config.OUTPUT_PATH
                / EXP_STRATEGY.name
                / EXP_DATASET.name
                / "weighted_f1-score.csv.gz"
            )
            if METRIC_RESULTS_FILE.exists():
                original_df = pd.read_csv(METRIC_RESULTS_FILE)
                print(original_df)


def acc_auc(config: Config):
    pass


for computed_metric in config.METRICS:
    computed_metric_class = getattr(
        importlib.import_module("metrics.computed." + computed_metric),
        computed_metric,
    )
    computed_metric_class = computed_metric_class(config)
    computed_metric_class.compute()
