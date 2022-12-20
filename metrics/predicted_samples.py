from framework_runners.base_runner import AL_Experiment
from metrics.base_metric import Base_Metric


# TODO ersetzt pickled learner, ersetzt
class Predicted_Samples(Base_Metric):
    metrics = ["y_pred_train", "y_pred_test"]

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass
