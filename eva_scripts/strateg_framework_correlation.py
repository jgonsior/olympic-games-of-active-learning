from collections import defaultdict
import sys
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt, ticker
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
from pathlib import Path

from misc.helpers import (
    _calculate_fig_size,
    create_fingerprint_joined_timeseries_csv_files,
    log_and_time,
    save_correlation_plot,
)
from misc.plotting import set_seaborn_style
import seaborn as sns

from misc.helpers import _calculate_fig_size
from misc.plotting import set_matplotlib_size, set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()

df = pd.read_parquet(
    "/home/jg/exp_results/full_exp_jan/plots/single_hyperparameter/EXP_STRATEGY/single_hyper_EXP_STRATEGY_full_auc_weighted_f1-score.parquet"
)
print(df)

framework_dict = defaultdict(list)
framework_dict_other = defaultdict(list)
framework_strats = defaultdict(list)


fra_to_fra_corr = []

for ix_a, row in df.iterrows():
    for ix_b, c in row.items():
        if not "(" in ix_a or not "(" in ix_b:
            continue

        ending_a = ix_a.split("(")[1].removesuffix(")")
        ending_b = ix_b.split("(")[1].removesuffix(")")

        fra_to_fra_corr.append([ending_a, ending_b, c])
        if ending_a == ending_b:
            framework_dict[ending_a].append(c)
            framework_strats[ending_a].append(ix_a)
        else:
            framework_dict_other[ending_a].append(c)

for k, v in framework_dict_other.items():
    print(f"{k}: {np.mean(v):.2f}+-{np.std(v):.2f}, {set(framework_strats[k])}")

df = pd.DataFrame(fra_to_fra_corr, columns=["fra", "frb", "corr"])

df = df["corr"].groupby([df["fra"], df["frb"]]).apply(np.mean).reset_index()
df = df.pivot(index="fra", columns="frb", values="corr")
# df.columns = df.columns.droplevel(level=0)
print(df.to_numpy())
print(df.columns)
set_seaborn_style(font_size=5)
save_correlation_plot(
    data=df.to_numpy(),
    title="framework_al_strat_correlation",
    keys=df.columns.str.upper(),
    config=config,
    total=False,
    # rotation=30,
)
