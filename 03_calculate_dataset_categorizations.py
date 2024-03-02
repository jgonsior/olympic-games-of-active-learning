import itertools
import sys
from typing import Tuple



sys.dont_write_bytecode = True

from misc.config import Config
from resources.data_types import (
    SAMPLES_CATEGORIZER,
    samples_categorizer_to_classes_mapping,
)
from metrics.computed.base_samples_categorizer import *

from misc.helpers import (
    create_workload,
    prepare_eva_pathes,
    run_from_workload,
)

config = Config()


prepare_eva_pathes("DATASET_CATEGORIZATIONS", config)


if config.EVA_MODE == "create":

    if config.SAMPLES_CATEGORIZER == ["_ALL"]:
        config.SAMPLES_CATEGORIZER = [sc.name for sc in SAMPLES_CATEGORIZER]

    print(
        "computung the following samples categories: "
        + ",".join(config.SAMPLES_CATEGORIZER)
    )

    dataset_categorizer_combinations: List[Tuple[DATASET, SAMPLES_CATEGORIZER]] = list(
        itertools.product(config.SAMPLES_CATEGORIZER, config.EXP_GRID_DATASET)
    )

    create_workload(
        dataset_categorizer_combinations,
        config=config,
        SLURM_ITERATIONS_PER_BATCH=1,
        SCRIPTS_PATH="dataset_categorizations",
        SLURM_NR_THREADS=1,
        script_type="dataset_categorization",
    )
elif config.EVA_MODE in ["local", "slurm", "single"]:
    # load parquet folder line?
    def _run_samples_categorizer(sc: str, ds: DATASET, config: Config):
        ds = DATASET(ds)
        print("#" * 100)
        print("computed_metric: " + str(sc) + " " + ds.name)
        samples_categorizer_class = samples_categorizer_to_classes_mapping[
            SAMPLES_CATEGORIZER[sc]
        ](config)

        samples_categorizer_class.categorize_samples(ds)

    run_from_workload(do_stuff=_run_samples_categorizer, config=config)
