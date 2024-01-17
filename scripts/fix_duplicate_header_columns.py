import multiprocessing
from pathlib import Path
import sys
import glob
import lzma

from joblib import Parallel, delayed

sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import shutil

pandarallel.initialize(progress_bar=True)
config = Config()


glob_list = [
    f for f in glob.glob(str(config.OUTPUT_PATH) + "/**/*.csv.xz", recursive=True) if  not f.endswith("_workload.csv.xz") and not f.endswith("_workloads.csv.xz")
]



def _do_stuff(file_name):  
    metric_file = Path(file_name)
    tmp_metric_file = Path(str(metric_file) + ".tmp")

    with lzma.open(metric_file, "rt") as mf:
        with lzma.open(tmp_metric_file, "wt") as tmf:
            for ix, line in enumerate(mf):
                if ix == 0 or "EXP_UNIQUE_ID" not in line:
                    tmf.write(line)
                else:
                    print(metric_file)
                if ix > 1000:
                    break
    shutil.move(src=tmp_metric_file, dst=metric_file)


Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(
#Parallel(n_jobs=1, verbose=10)(
    delayed(_do_stuff)(file_name) for file_name in glob_list
)
