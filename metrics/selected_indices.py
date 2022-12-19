from framework_runners.base_runner import AL_Experiment
from metrics.base_metric import Base_Metric


class Selected_Indice(Base_Metric):
    metrics = ["selected_indice"]

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass
