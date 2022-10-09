import itertools
from typing import Any, Dict, List

from resources.data_types import AL_STRATEGY
from misc.config import Config


def _create_exp_grid(
    exp_strat_grid: List[Dict[AL_STRATEGY, Dict[str, List[Any]]]], config: Config
) -> List[str]:
    result: List[str] = []
    for a in exp_strat_grid:
        for b, c in a.items():
            kwargs = []
            for d, e in c.items():
                kwargs.append(
                    [f"{d}{config._EXP_STRATEGY_PARAM_VALUE_DELIM}{_x}" for _x in e]
                )
            for f in [
                config._EXP_STRATEGY_PARAM_PARAM_DELIM.join(_x)
                for _x in itertools.product(*kwargs)
            ]:
                result.append(f"{b}{config._EXP_STRATEGY_STRAT_PARAMS_DELIM}{f}")
    return result
