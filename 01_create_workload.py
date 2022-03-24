# read in which datasets and strategies should be compared and what experiment settings
# check if in results.csv the results are already existeng
# if not, generate workload csv

from misc.config import Config
from misc.logging import log_it


cfg = Config()

print(cfg.RANDOM_SEED)
log_it("random_seed")
