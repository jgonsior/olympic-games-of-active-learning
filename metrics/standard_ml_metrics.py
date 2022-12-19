from framework_runners.base_runner import AL_Experiment
from metrics.base_metric import Base_Metric


class Standard_ML_Metrics(Base_Metric):
    metrics = [
        "accuracy",
        "macro_f1-score",
        "weighted_f1-score",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "macro_precision",
    ]

    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass
