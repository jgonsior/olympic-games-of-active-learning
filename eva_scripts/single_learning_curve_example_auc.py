import sys
from turtle import up
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

"""[
            2.4,
            4,
            5,
            4.8,
            5,
            5.8,
            6,
            6,
            6.4,
            6,  ##
            5.8,
            6,
            6,
            6.4,
            6,
            5.8,
            6,
            5.8,
            6,
            6,
            6.4,
            6,
            5.8,
            6,
            5.8,
            6,
            6,
            6.4,
            6,
            5.8,
            6,
        ],"""

data = pd.DataFrame(
    data=[[*[iii for iii in range(0, 100)], *[100 for _ in range(0, 100)]]],
).T
# data /= 8
print(data)
current_row_np = data[0].to_numpy()
print(current_row_np)

average_improvement_over_all_time_steps = np.sum(np.diff(current_row_np)) / len(
    current_row_np
)

window_size = 10
cutoff_value = None
for window_param in range(window_size, len(current_row_np)):
    fixed_window = current_row_np[
        (len(current_row_np) - window_param) : (
            len(current_row_np) - window_param + window_size
        )
    ]
    print(fixed_window)

    fixed_average = np.sum(np.diff(fixed_window)) / len(fixed_window)
    print(
        f"{len(current_row_np)-window_param}: {fixed_average:.2} >? {average_improvement_over_all_time_steps}  ({np.diff(fixed_window)})"
    )
    if fixed_average > average_improvement_over_all_time_steps:
        cutoff_value = len(current_row_np) - window_param + 1
        break

print(cutoff_value)

if cutoff_value is None:
    cutoff_value = round(len(current_row_np) / 2)
print(cutoff_value)


set_seaborn_style(font_size=7)
sns.set_style("whitegrid")

plt.figure(figsize=_calculate_fig_size(3.57))
ax = sns.lineplot(data, legend=False, markers=True)
ax.axvline(cutoff_value, color="r")
ax.set(title="", xlabel="AL Cycle", ylabel="ML Performance Metric")

# ax.xaxis.set_major_locator(ticker.FixedLocator([rrr for rrr in range(0, 10)]))
# ax.set_title(f"Learning Curve: {standard_metric}")

aucs = [
    ("full auc", 0, 9),
    ("plateau", 5, 4),
    ("first 5", 0, 5),
    ("last 5", 5, 5),
    ("ramp-up", 0, 5),
    ("last value", 9, 1),
]

"""for ix, auc in enumerate(aucs):
    ax.arrow(
        x=auc[1],
        dx=auc[2] + 0.2,
        y=ix * 0.2,
        dy=0,
        shape="full",
        width=0.07,
        length_includes_head=True,
        color=sns.color_palette()[ix + 1],
    )

    ax.arrow(
        x=auc[2] - auc[1],
        dx=-auc[2] - 0.2,
        y=ix * 0.2,
        dy=0,
        shape="full",
        width=0.07,
        length_includes_head=True,
        color=sns.color_palette()[ix + 1],
    )

    ax.text(
        x=auc[1],
        y=ix * 0.2,
        s=auc[0],
        ha="left",
        va="center",
        rotation=0,
        size=6,
        color="white",
    )"""

plt.legend([], [], frameon=False)


destination_path = Path(config.OUTPUT_PATH / f"plots/single_learning_curve/")
data.to_parquet(destination_path / f"single_exemplary_learning_curve_auc.parquet")
plt.savefig(
    destination_path / f"single_exemplary_learning_curve_auc.pdf",
    dpi=3000,
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    destination_path / f"single_exemplary_learning_curve_auc.jpg",
    dpi=3000,
    bbox_inches="tight",
    pad_inches=0,
)
# goal: dataframe where each column is an EXP_STRATEGY and each row is a DATASET --> rest is aggregated over all params
