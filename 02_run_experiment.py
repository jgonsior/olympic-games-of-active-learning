from framework_runners.alipy_runner import ALIPY_AL_Experiment
from framework_runners.base_runner import AL_Experiment
from framework_runners.optimal_runner import OPTIMAL_AL_Experiment
from framework_runners.libact_runner import LIBACT_Experiment
from framework_runners.playground_runner import PLAYGROUND_AL_Experiment
from misc.config import Config
from resources.data_types import AL_FRAMEWORK, AL_STRATEGY


config = Config()

al_experiment: AL_Experiment
print(str(config.EXP_STRATEGY)[12:])
if str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.ALIPY.name)):
    al_experiment = ALIPY_AL_Experiment(config)
elif str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.OPTIMAL.name)):
    al_experiment = OPTIMAL_AL_Experiment(config)
elif str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.LIBACT.name)):
    al_experiment = LIBACT_Experiment(config)
elif str(config.EXP_STRATEGY)[12:].startswith(str(AL_FRAMEWORK.PLAYGROUND.name)):
    al_experiment = PLAYGROUND_AL_Experiment(config)
else:
    raise ValueError(
        "Error, could not find the specified AL Strategy or the framework does not exist"
    )


al_strategies_to_temporarily_ignore = [
    AL_STRATEGY.ALIPY_BMDR,
    AL_STRATEGY.ALIPY_CORESET_GREEDY,
    AL_STRATEGY.OPTIMAL_GREEDY,
    AL_STRATEGY.ALIPY_LAL,
    AL_STRATEGY.ALIPY_SPAL,
    AL_STRATEGY.ALIPY_UNCERTAINTY_DTB,
    AL_STRATEGY.ALIPY_QBC,
    AL_STRATEGY.ALIPY_EXPECTED_ERROR_REDUCTION,
]

if config.EXP_STRATEGY in al_strategies_to_temporarily_ignore:
    exit(-1)


al_experiment.run_experiment()
