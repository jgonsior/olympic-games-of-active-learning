from enum import IntEnum, unique
from typing import Dict
from analyse_results.visualizations.auc_table import Auc_Table
from analyse_results.visualizations.base_visualizer import Base_Visualizer
from analyse_results.visualizations.learning_curves import Learning_Curves
from analyse_results.visualizations.retrieved_samples import (
    Retrieved_Samples,
)

from analyse_results.visualizations.run_done_stats_table import (
    Run_Done_Stats_Table,
)
from analyse_results.visualizations.runtimes import Runtimes
from analyse_results.visualizations.strategy_ranking import Strategy_Ranking


@unique
class VISUALIZATION(IntEnum):
    LEARNING_CURVES = 1
    RETRIEVED_SAMPLES = 2
    RUN_DONE_STATS = 3
    STRATEGY_RANKING = 4
    RUNTIMES = 5
    AUC_TABLE = 6


vizualization_to_python_function_mapping: Dict[VISUALIZATION, type[Base_Visualizer]] = {
    VISUALIZATION.LEARNING_CURVES: Learning_Curves,
    VISUALIZATION.RETRIEVED_SAMPLES: Retrieved_Samples,
    VISUALIZATION.RUN_DONE_STATS: Run_Done_Stats_Table,
    VISUALIZATION.STRATEGY_RANKING: Strategy_Ranking,
    VISUALIZATION.RUNTIMES: Runtimes,
    VISUALIZATION.AUC_TABLE: Auc_Table,
}
