from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, List

from typing import Any, Dict

if TYPE_CHECKING:
    from misc.config import Config


class Base_Visualizer(ABC):
    def __init__(self, config: Config) -> None:
        self._config = config

    @abstractproperty
    def template_name(self) -> str:
        ...

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        ...
