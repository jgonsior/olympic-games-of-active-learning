from __future__ import annotations

from interactive_results_browser.visualizations.base import Base_Visualizer
from typing import TYPE_CHECKING, Any, List, Tuple

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Strategy_Ranking(Base_Visualizer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def template_name(self) -> str:
        return "strategy_ranking.html.j2"

    def get_data(self) -> Dict[str, Any]:
        return super().get_data()
