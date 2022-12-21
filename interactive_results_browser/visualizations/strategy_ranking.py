from __future__ import annotations

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import Any, TYPE_CHECKING

from typing import Any, Dict

if TYPE_CHECKING:
    pass


class Strategy_Ranking(Base_Visualizer):
    def get_template_data(self) -> Dict[str, Any]:
        return super().get_template_data()
