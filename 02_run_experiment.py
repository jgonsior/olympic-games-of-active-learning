from framework_runners.alipy import ALIPY_AL_Experiment
from framework_runners.base_runner import AL_Experiment
from framework_runners.optimal import OPTIMAL_AL_Experiment
from misc.config import Config
from ressources.data_types import AL_FRAMEWORK


config = Config()

al_experiment: AL_Experiment
if str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.ALIPY.name)):
    al_experiment = ALIPY_AL_Experiment(config)
elif str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.OPTIMAL.name)):
    al_experiment = OPTIMAL_AL_Experiment(config)
else:
    raise ValueError(
        "Error, could not find the specified AL Strategy or the framework does not exist"
    )

al_experiment.run_experiment()
