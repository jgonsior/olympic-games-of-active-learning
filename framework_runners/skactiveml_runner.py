from __future__ import annotations

from typing import TYPE_CHECKING
from framework_runners.base_runner import AL_Experiment

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
    )

class SKACTIVEML_AL_Experiment(AL_Experiment):
    def __init__(self, config: "Config"):
        super().__init__(config)
        self.al_strategy = None

    def get_AL_strategy(self):
        pass

    def query_AL_strategy(self) -> SampleIndiceList:
        return None

    def prepare_dataset(self):
        pass