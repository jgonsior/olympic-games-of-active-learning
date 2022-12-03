from __future__ import annotations

from interactive_results_browser.visualizations.base_visualizer import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Retrieved_Samples(Base_Visualizer):
    def get_template_data(self) -> Dict[str, Any]:
        return super().get_template_data()
