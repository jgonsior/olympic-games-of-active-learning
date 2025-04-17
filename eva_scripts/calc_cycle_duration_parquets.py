import sys
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
from pathlib import Path
from pyarrow.parquet import ParquetFile
import pyarrow as pa


from misc.helpers import _calculate_fig_size
from misc.plotting import set_matplotlib_size, set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

# df = pd.read_parquet(config.CORRELATION_TS_PATH / "weighted_f1-score.parquet")


pf = ParquetFile(config.CORRELATION_TS_PATH / "weighted_f1-score.parquet")
first_ten_rows = next(pf.iter_batches(batch_size=100))
df = pa.Table.from_batches([first_ten_rows]).to_pandas()


print(df)
