import csv
import glob
from typing import Dict, List, Optional
import pandas as pd
from pyparsing import Path

from misc.config import Config


def _append_and_create(file_name: Path, content: Dict):
    if not file_name.exists():
        with open(file_name, "w") as f:
            w = csv.DictWriter(f, fieldnames=content.keys())
            w.writeheader()

    with open(file_name, "a") as f:
        w = csv.DictWriter(f, fieldnames=content.keys())
        w.writerow(content)


# read in csv
# in case of errors -> skip file and return None
def _get_df(file_name: Path, config: Config) -> Optional[pd.DataFrame]:
    print(file_name)
    try:
        if file_name.name.endswith(".csv.xz") or file_name.name.endswith(".csv"):
            df = pd.read_csv(file_name)
        else:
            df = pd.read_parquet(file_name)
    except Exception as err:
        error_message = f"ERROR: {err.__class__.__name__} - {err.args}"
        print(error_message)

        _append_and_create(
            config.BROKEN_CSV_FILE_PATH,
            {"metric_file": file_name, "error_message": error_message},
        )

        return None

    return df


def _get_glob_list(
    config: Config,
    limit: str = "**",
    ignore_original_workloads=True,
) -> List[Path]:
    glob_list = [
        *[
            ggg
            for ggg in glob.glob(
                str(config.OUTPUT_PATH) + f"/{limit}/*.csv.xz", recursive=True
            )
        ],
        *[
            ggg
            for ggg in glob.glob(
                str(config.OUTPUT_PATH) + f"/{limit}/*.csv.xz.parquet", recursive=True
            )
        ],
    ]

    if ignore_original_workloads:
        glob_list = [
            ggg
            for ggg in glob_list
            if not ggg.endswith("_workload.csv.xz")
            and not ggg.endswith("_workloads.csv.xz")
        ]

    glob_list = [Path(ggg) for ggg in glob_list]
    return sorted(glob_list)
