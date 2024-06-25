import sys

import dask.dataframe as dd


sys.dont_write_bytecode = True

from misc.config import Config

# all batches which have been running longer than 10 minutes will be ignored

config = Config()

for EXP_DATASET in config.EXP_GRID_DATASET:
    print(EXP_DATASET.name)

    a = dd.read_csv(
        f"{config.DATASETS_PATH}/{EXP_DATASET.name}{config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}"
    )
    a.to_parquet(
        f"{config.DATASETS_PATH}/{EXP_DATASET.name}{config.DATASETS_TRAIN_TEST_SPLIT_APPENDIX}.parquet"
    )
    b = dd.read_csv(
        f"{config.DATASETS_PATH}/{EXP_DATASET.name}.csv",
    )
    b.to_parquet(f"{config.DATASETS_PATH}/{EXP_DATASET.name}.parquet")
    c = dd.read_csv(
        f"{config.DATASETS_PATH}/{EXP_DATASET.name}{config.DATASETS_DISTANCES_APPENDIX}",
    )
    c.to_parquet(
        f"{config.DATASETS_PATH}/{EXP_DATASET.name}{config.DATASETS_DISTANCES_APPENDIX}.parquet"
    )
