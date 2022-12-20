from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import importlib
import random
from typing import TYPE_CHECKING, Any, List
from datasets import DATASET, load_dataset, split_dataset
from metrics.base_metric import Base_Metric
from misc.logging import log_it

if TYPE_CHECKING:
    from misc.config import Config
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning


class AL_Experiment(ABC):
    config: Config
    select_ind: List[int]
    metrics: List[Base_Metric] = []
    y_pred_train_calculated: bool = False
    y_pred_test_calculated: bool = False

    def __init__(self, config: Config) -> None:
        self.config = config

        for metric_class in config.METRICS:
            metric_class = getattr(
                importlib.import_module("metrics." + metric_class), metric_class
            )
            self.metrics.append(metric_class())

        log_it(
            f"Executing Job # {self.config.WORKER_INDEX} of workload {self.config.WORKLOAD_FILE_PATH}: {self.config.EXP_DATASET.name} {self.config.EXP_STRATEGY.name}"
        )

    @abstractmethod
    def get_AL_strategy(self):
        pass

    @abstractmethod
    def query_AL_strategy(self) -> List[int]:
        pass

    @abstractmethod
    def prepare_dataset(self):
        pass

    def al_cycle(self, iteration_counter: int) -> None:
        log_it(f"#{iteration_counter}")

        self.y_pred_train_calculated = False
        self.y_pred_test_calculated = False

        for metric in self.metrics:
            metric.pre_query_selection_hook(self)

        # only use the query strategy if there are actualy samples left to label
        if iteration_counter == 0:
            # "fake" iteration zero
            self.select_ind = self.labeled_idx
            self.labeled_idx = []
        elif len(self.unlabeled_idx) > self.config.EXP_BATCH_SIZE:
            self.select_ind = self.query_AL_strategy()
        else:
            # if we have labeled everything except for a small batch -> return that
            self.select_ind = self.unlabeled_idx

        for metric in self.metrics:
            metric.post_query_selection_hook(self)

        self.labeled_idx = self.labeled_idx + self.select_ind
        self.unlabeled_idx = self._list_difference(self.unlabeled_idx, self.select_ind)

        for metric in self.metrics:
            metric.pre_retraining_of_learner_hook(self)

        # update our learner model
        self.model.fit(X=self.X[self.labeled_idx, :], y=self.Y[self.labeled_idx])  # type: ignore

        for metric in self.metrics:
            metric.post_retraining_of_learner_hook(self)

    def run_experiment(self) -> None:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        np.random.seed(self.config.EXP_RANDOM_SEED)
        random.seed(self.config.EXP_RANDOM_SEED)

        dataset = DATASET(self.config.EXP_DATASET)
        dataset_tuple = load_dataset(dataset, self.config)

        # load dataset
        (
            self.X,
            self.Y,
            self.train_idx,
            self.test_idx,
            self.labeled_idx,
            self.unlabeled_idx,
        ) = split_dataset(dataset_tuple, self.config)

        self.prepare_dataset()

        from resources.data_types import (
            learner_models_to_classes_mapping,
        )

        # load ml model
        model_instantiation_tuple = learner_models_to_classes_mapping[
            self.config.EXP_LEARNER_MODEL
        ]
        self.model = model_instantiation_tuple[0](**model_instantiation_tuple[1])

        # either we stop until all samples are labeled, or earlier
        if self.config.EXP_NUM_QUERIES == 0:
            total_amount_of_iterations = (
                int(len(self.unlabeled_idx) / self.config.EXP_BATCH_SIZE) + 1
            )
        else:
            total_amount_of_iterations = self.config.EXP_NUM_QUERIES

        # select the AL strategy to use
        self.get_AL_strategy()

        for iteration in range(0, total_amount_of_iterations):
            if len(self.unlabeled_idx) == 0:
                log_it("early stopping")
                break

            self.al_cycle(iteration_counter=iteration)

        for metric in self.metrics:
            metric.save_metrics(self)

        # global results
        with open(self.config.OVERALL_DONE_WORKLOAD_PATH, "a") as f:
            w = csv.DictWriter(f, fieldnames=self.config._original_workload.keys())

            if self.config.OVERALL_DONE_WORKLOAD_PATH.stat().st_size == 0:
                w.writeheader()
            w.writerow(self.config._original_workload)

    # efficient list difference
    def _list_difference(
        self, long_list: List[Any], short_list: List[Any]
    ) -> List[Any]:
        short_set = set(short_list)
        return [i for i in long_list if not i in short_set]

    def get_y_pred_train(self) -> List[int]:
        if not self.y_pred_train_calculated:
            self.y_pred_train = self.model.predict(self.X[self.train_idx, :]).tolist()
            self.y_pred_train_calculated = True
        return self.y_pred_train

    def get_y_pred_test(self) -> List[int]:
        if not self.y_pred_test_calculated:
            self.y_pred_test = self.model.predict(self.X[self.test_idx, :]).tolist()
            self.y_pred_test_calculated = True
        return self.y_pred_test
