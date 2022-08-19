from __future__ import annotations
from pathlib import Path
from typing import List

from typing import TYPE_CHECKING, Any, List

import yaml

if TYPE_CHECKING:
    from misc.config import Config


def get_exp_config_names(config: Config) -> List[str]:
    yaml_config_params = yaml.safe_load(Path(config.LOCAL_YAML_EXP_PATH).read_bytes())
    return yaml_config_params.keys()
