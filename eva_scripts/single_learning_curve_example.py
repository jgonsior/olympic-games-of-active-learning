import sys
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd
from pathlib import Path


from misc.helpers import _calculate_fig_size
from misc.plotting import set_matplotlib_size, set_seaborn_style
import seaborn as sns

sys.dont_write_bytecode = True

from misc.config import Config

config = Config()


data = pd.DataFrame(
    data=[
        [2.4, 4, 5, 4.8, 5, 5.8, 6, 6, 6.4, 6],
        [2, 2, 3, 4, 6, 6, 6.4, 7, 8, 7],
        [6.425 * 8 / 10 for _ in range(0, 10)],
    ],
    index=["Strategy A", "Strategy B", "Strategy C"],
).T
data /= 8

print("mean")
print(data["Strategy A"].mean())
print(data["Strategy B"].mean())
print(data["Strategy C"].mean())

print("trapz")
print(np.trapz(data["Strategy A"], dx=1))
print(np.trapz(data["Strategy B"], dx=1))
print(np.trapz(data["Strategy C"], dx=1))

print("auc/max_auc")
print(np.trapz(data["Strategy A"], dx=1) / np.trapz([1 for _ in range(0, 10)], dx=1))
print(np.trapz(data["Strategy B"], dx=1) / np.trapz([1 for _ in range(0, 10)], dx=1))
print(np.trapz(data["Strategy C"], dx=1) / np.trapz([1 for _ in range(0, 10)], dx=1))


set_seaborn_style(font_size=7, usetex=True)
# plt.figure(figsize=set_matplotlib_size(fraction=10))

# calculate fraction based on length of keys
plt.figure(figsize=_calculate_fig_size(3.57))
ax = sns.lineplot(data)

ax.set(title="", xlabel="AL Cycle", ylabel="ML Performance Metric")

ax.xaxis.set_major_locator(ticker.FixedLocator([rrr for rrr in range(0, 10)]))
# ax.set_title(f"Learning Curve: {standard_metric}")


destination_path = Path(config.OUTPUT_PATH / f"plots/single_learning_curve/")
ts.to_parquet(destination_path / f"single_exemplary_learning_curve.parquet")
plt.savefig(
    destination_path / f"single_exemplary_learning_curve.pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
)
# goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
