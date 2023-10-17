from pathlib import Path
import sys
from typing import List

from matplotlib import pyplot as plt
from analyse_results.helper_functions import (
    _download_static_ressources,
    get_exp_grid_without_flask_params,
    render_template,
)
from analyse_results.visualizations.base_visualizer import Base_Visualizer
from typing import List
from enum import Enum
from analyse_results.visualizations import (
    vizualization_to_python_function_mapping,
)
from jinja2 import Template
from analyse_results.visualizations.base_visualizer import Base_Visualizer
from misc.config import Config
from collections.abc import Iterable

import matplotlib.pyplot as plt


sys.dont_write_bytecode = True

from misc.config import Config
from pandarallel import pandarallel
import pandas as pd

pandarallel.initialize(progress_bar=True)
import sys
from datasets import DATASET
from resources.data_types import AL_STRATEGY

# pd.options.display.float_format = "{:100,.2f}".format

config = Config()

_download_static_ressources()

# @TODO das hier irgendwie anders definieren per CLI params oder json string input oder was weiß ich
exp_grid_request_params, full_exp_grid = get_exp_grid_without_flask_params(config)

print(exp_grid_request_params)
print("\n" * 3)
print(full_exp_grid)
print("\n" * 3)


visualizations_and_tables: List[Base_Visualizer] = []

plt.ioff()

for viz in exp_grid_request_params["VISUALIZATIONS"]:
    visualizer = vizualization_to_python_function_mapping[viz](
        config, exp_grid_request_params, config.EXP_TITLE
    )
    visualizations_and_tables.append(visualizer)

print(visualizations_and_tables)


for viz in visualizations_and_tables:
    additional_request_params = viz.get_additional_request_params(config.OUTPUT_PATH)
    if additional_request_params != {}:
        full_exp_grid = {
            **full_exp_grid,
            **additional_request_params,
        }

rendered_template = render_template(
    "show_interactive_viz.html.j2",
    experiment_name=config.EXP_TITLE,
    config=config,
    full_exp_grid=full_exp_grid,
    visualizations_and_tables=visualizations_and_tables,
    isinstance=isinstance,
    Iterable=Iterable,
    Enum=Enum,
    int=int,
    str=str,
    exp_grid_request_params=exp_grid_request_params,
)

print(rendered_template)
