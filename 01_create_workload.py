# read in which datasets and strategies should be compared and what experiment settings
# check if in results.csv the results are already existeng
# if not, generate workload csv

from misc.config import Config
from misc.logging import log_it


cfg = Config()

print(cfg.RANDOM_SEED)
print(cfg.LEARNER_ML_MODEL)
print(cfg.HPC_DATASET_PATH)
log_it(cfg)
