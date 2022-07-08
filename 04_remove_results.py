import ast
from itertools import chain
import multiprocessing
import timeit
import pandas as pd
import csv
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import pandas as pd
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed
from ressources.data_types import AL_STRATEGY
from joblib import Parallel, delayed

config = Config()


# done_workload = pd.read_csv(    str(config.EXP_RESULT_EXTRACTED_ZIP_PATH / config.DONE_WORKLOAD_PATH.name))
done_workload = pd.read_csv(config.DONE_WORKLOAD_FILE)


done_workload["EXP_FULL_STRATEGY"] = (
    done_workload["EXP_STRATEGY"] + "#" + done_workload["EXP_STRATEGY_PARAMS"]
)

remove_quire = False

if remove_quire:
    # remove quire
    print(done_workload)
    quire_workload = done_workload.loc[
        done_workload["EXP_STRATEGY"] == "AL_STRATEGY.ALIPY_QUIRE"
    ]
    quire_workload.to_csv(str(config.DONE_WORKLOAD_FILE) + "_quire.csv", index=None)
    done_workload = done_workload.loc[
        ~(done_workload["EXP_STRATEGY"] == "AL_STRATEGY.ALIPY_QUIRE")
    ]
    done_workload.to_csv(config.DONE_WORKLOAD_FILE, index=None)
    print(len(done_workload))

else:
    quire_workload = pd.read_csv(
        str(config.DONE_WORKLOAD_FILE) + "_quire.csv", nrows=100
    )
    print(quire_workload["EXP_UNIQUE_ID"])
