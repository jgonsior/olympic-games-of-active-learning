from __future__ import annotations
from typing import List

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from misc.config import Config


def get_exp_config_names(config: Config) -> List[str]:
    return ["hui", "hoi", "hallo", "taddl"]
