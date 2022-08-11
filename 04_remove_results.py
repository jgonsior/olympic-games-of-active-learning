import ast
from itertools import chain
import multiprocessing
import timeit
import modin.pandas as pd
import csv
from pathlib import Path
import stat
from typing import Any, Dict, List
from jinja2 import Template
import modin.pandas as pd
from misc.config import Config
from misc.logging import log_it
from sklearn.model_selection import ParameterGrid
import os
from joblib import Parallel, delayed
from resources.data_types import AL_STRATEGY
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
        str(config.DONE_WORKLOAD_FILE) + "_quire.csv",
    )
    for _, row in quire_workload.iterrows():
        dataset = row["EXP_DATASET"]
        unique_id = row["EXP_UNIQUE_ID"]
        dataset = dataset.replace("DATASET.", "")
        metric_results = Path(
            f"{config.RESULTS_PATH}/{dataset}/{unique_id}_metric_results.csv"
        )
        #  print(f"delete {metric_results}")
        if metric_results.exists():
            metric_results.unlink()
            print(f"delete {metric_results}")
        else:
            print(f"not found {metric_results}")
