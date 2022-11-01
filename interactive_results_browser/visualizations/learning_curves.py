from __future__ import annotations

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Learning_Curves(Base_Visualizer):
    def get_template_data(self) -> Dict[str, Any]:
        # read in all metrics
        # plotly
        # group by irgendwas
        # let metric of learning plots be determined by get parameter list
        return super().get_template_data()
