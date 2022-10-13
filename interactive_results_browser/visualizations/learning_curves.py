from __future__ import annotations

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Learning_Curves(Base_Visualizer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_data(self) -> Dict[str, Any]:
        return super().get_data()

    def template_name(self) -> str:
        return "learning_curves.html.j2"
