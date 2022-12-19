from abc import ABC, abstractmethod

from framework_runners.base_runner import AL_Experiment


class Base_Metric(ABC):
    @abstractmethod
    def pre_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def post_retraining_of_learner_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def pre_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    @abstractmethod
    def post_query_selection_hook(self, al_experiment: AL_Experiment):
        pass

    def learner_training_time(self):
        ...

    def query_selection_time(self):
        ...

    def accuracy(self):
        ...

    def acc_auc(self):
        ...

    def macro_f1_auc(self):
        ...

    def macro_prec_auc(self):
        ...

    def macro_recall_auc(self):
        ...

    def weighted_f1_auc(self):
        ...

    def weighted_prec_auc(self):
        ...

    def weighted_recall_auc(self):
        ...

    def selected_indices(self):
        ...

    def pickled_learner_model(self):
        ...
