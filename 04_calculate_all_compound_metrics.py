import sys

sys.dont_write_bytecode = True
import importlib

from misc.config import Config
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

config = Config()

print("computung the following metrics: " + ",".join(config.COMPUTED_METRICS))


for samples_categorizer in config.SAMPLES_CATEGORIZER:
    print("#" * 100)
    print("computed_metric: " + str(samples_categorizer))
    samples_categorizer_class = getattr(
        importlib.import_module("metrics.computed." + samples_categorizer),
        samples_categorizer,
    )
    samples_categorizer_class = samples_categorizer_class(config)

    samples_categorizer_class.categorize_samples()


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
