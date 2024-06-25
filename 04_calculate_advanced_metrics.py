import itertools
import sys
from typing import List, Tuple


from datasets import DATASET
from resources.data_types import AL_STRATEGY, COMPUTED_METRIC

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from misc.helpers import (
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
)

config = Config()


prepare_eva_pathes("advanced_metrics", config)


if config.EVA_MODE == "create":
    if config.COMPUTED_METRICS == ["_ALL"]:
        # ensure that STANDARD_AUC is run last so that all other metrics are already precomputed
        config.COMPUTED_METRICS = [
            sc.name
            for sc in COMPUTED_METRIC
            if sc != COMPUTED_METRIC.STANDARD_AUC
            and sc != COMPUTED_METRIC.TIMELAG_METRIC
        ] + [COMPUTED_METRIC.TIMELAG_METRIC.name, COMPUTED_METRIC.STANDARD_AUC.name]

    print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))

    print(config.COMPUTED_METRICS)

    metric_combinations: List[Tuple[COMPUTED_METRIC, AL_STRATEGY, DATASET]] = list(
        itertools.product(
            config.COMPUTED_METRICS,
            config.EXP_GRID_STRATEGY,
            config.EXP_GRID_DATASET,
        )
    )

    create_workload(
        metric_combinations,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1,
        SCRIPTS_PATH="metrics",
        SLURM_NR_THREADS=1,
        script_type="metrics",
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:

    def _run_single_metric(
        computed_metric: COMPUTED_METRIC,
        exp_strategy: AL_STRATEGY,
        exp_dataset: DATASET,
        config: Config,
    ):
        exp_strategy = AL_STRATEGY(exp_strategy)
        exp_dataset = DATASET(exp_dataset)
        print(computed_metric)
        print(exp_strategy.name)
        print(exp_dataset.name)

        computed_metric_class = getattr(
            importlib.import_module("metrics.computed." + computed_metric),
            computed_metric,
        )
        computed_metric_class = computed_metric_class(config)

        computed_metric_class.compute_metrics(
            exp_dataset=exp_dataset, exp_strategy=exp_strategy
        )

    run_from_workload(do_stuff=_run_single_metric, config=config)
