from framework_runners.base_runner import AL_Experiment
from metrics.base_metric import Base_Metric


class Pickled_Learner_Model(Base_Metric):
    metrics = ["pickled_learner_model"]

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass
