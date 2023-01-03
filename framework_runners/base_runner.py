from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import importlib
import random
from typing import TYPE_CHECKING, Any, Dict, List
from datasets import DATASET, load_dataset, split_dataset
from metrics.base_metric import Base_Metric
from misc.logging import log_it

if TYPE_CHECKING:
    from misc.config import Config

    from resources.data_types import (
        SampleIndiceList,
        FeatureVectors,
        LabelList,
    )

import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning


class AL_Experiment(ABC):
    config: Config
    metrics: List[Base_Metric] = []
    y_pred_train_calculated: bool = False
    y_pred_test_calculated: bool = False

    X: FeatureVectors
    Y: LabelList
    global_train_idx: SampleIndiceList
    global_test_idx: SampleIndiceList
    global_initially_labeled_idx: SampleIndiceList

    map_global_to_local_train_ix: Dict[int, int]
    map_local_to_global_train_ix: Dict[int, int]

    local_X_train: SampleIndiceList
    local_Y_train: SampleIndiceList
    local_train_labeled_idx: SampleIndiceList
    local_train_unlabeled_idx: SampleIndiceList

    local_selected_idx: SampleIndiceList

    def __init__(self, config: Config) -> None:
        self.config = config

        for metric_class in config.METRICS:
            metric_class = str(metric_class)
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
    def query_AL_strategy(self) -> SampleIndiceList:
        pass

    @abstractmethod
    def prepare_dataset(self):
        pass

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
            self.global_train_idx,
            self.global_test_idx,
            self.global_initially_labeled_idx,
        ) = split_dataset(dataset_tuple, self.config)

        # create better new indices
        self.local_X_train = self.X[self.global_train_idx]

        self.map_global_to_local_train_ix: Dict[int, int] = {
            global_ix: local_ix
            for local_ix, global_ix in enumerate(self.global_train_idx)
        }
        self.map_local_to_global_train_ix: Dict[int, int] = {
            local_ix: global_ix
            for global_ix, local_ix in self.map_global_to_local_train_ix.items()
        }
        self.local_Y_train = self.Y[self.global_train_idx]

        self.local_train_labeled_idx = [
            self.map_global_to_local_train_ix[ggg]
            for ggg in self.global_initially_labeled_idx
        ]
        self.local_train_unlabeled_idx = [
            lll
            for lll in self.map_local_to_global_train_ix.keys()
            if lll not in self.local_train_labeled_idx
        ]

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
                int(len(self.local_train_unlabeled_idx) / self.config.EXP_BATCH_SIZE)
                + 1
            )
        else:
            total_amount_of_iterations = self.config.EXP_NUM_QUERIES

        # select the AL strategy to use
        self.get_AL_strategy()

        for iteration in range(0, total_amount_of_iterations):
            if len(self.local_train_unlabeled_idx) == 0:
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

    def al_cycle(self, iteration_counter: int) -> None:
        log_it(f"#{iteration_counter}")

        self.y_pred_train_calculated = False
        self.y_pred_test_calculated = False

        for metric in self.metrics:
            metric.pre_query_selection_hook(self)

        # only use the query strategy if there are actualy samples left to label
        if iteration_counter == 0:
            # "fake" iteration zero
            self.local_selected_idx = self.local_train_labeled_idx
            self.local_train_labeled_idx = []
        elif len(self.local_train_unlabeled_idx) > self.config.EXP_BATCH_SIZE:
            self.local_selected_train_idx = self.query_AL_strategy()
            self.local_train_labeled_idx = (
                self.local_train_labeled_idx + self.local_selected_train_idx
            )
            self.local_train_unlabeled_idx = self._list_difference(
                self.local_train_unlabeled_idx, self.local_selected_train_idx
            )
        else:
            # if we have labeled everything except for a small batch -> return that
            self.local_selected_idx = self.local_train_unlabeled_idx

        local_select_idx_set = set(self.local_selected_idx)
        local_labeled_idx_set = set(self.local_train_labeled_idx)
        local_unlabeled_idx_set = set(self.local_train_unlabeled_idx)
        global_select_idx_set = set(
            [self.map_local_to_global_train_ix[lll] for lll in local_select_idx_set]
        )
        global_labeled_idx_set = set(
            [self.map_local_to_global_train_ix[lll] for lll in local_labeled_idx_set]
        )
        global_unlabeled_idx_set = set(
            [self.map_local_to_global_train_ix[lll] for lll in local_unlabeled_idx_set]
        )
        global_train_idx_set = set(self.global_train_idx)
        global_test_idx_set = set(self.global_test_idx)

        if iteration_counter > 0:
            print(local_select_idx_set)
            print(global_select_idx_set)

            print(local_labeled_idx_set)
            print(local_unlabeled_idx_set)
            print(global_labeled_idx_set)
            print(global_unlabeled_idx_set)

            print(global_train_idx_set)
            print(global_test_idx_set)

            assert local_select_idx_set.issubset(local_unlabeled_idx_set)
            assert global_select_idx_set.issubset(global_unlabeled_idx_set)

        # no duplicates
        assert len(self.local_selected_idx) == len(local_select_idx_set)

        assert global_select_idx_set.issubset(global_train_idx_set)
        assert len(global_select_idx_set.intersection(global_test_idx_set)) == 0
        assert len(global_select_idx_set.intersection(global_labeled_idx_set)) == 0
        assert global_select_idx_set.issubset(global_train_idx_set)

        for metric in self.metrics:
            metric.post_query_selection_hook(self)

        self.local_train_labeled_idx = (
            self.local_train_labeled_idx + self.local_selected_idx
        )
        self.local_train_unlabeled_idx = self._list_difference(
            self.local_train_unlabeled_idx, self.local_selected_idx
        )

        for metric in self.metrics:
            metric.pre_retraining_of_learner_hook(self)

        # update our learner model
        self.model.fit(X=self.local_X_train[self.local_train_labeled_idx, :], y=self.local_Y_train[self.local_train_labeled_idx])  # type: ignore

        for metric in self.metrics:
            metric.post_retraining_of_learner_hook(self)

    # efficient list difference
    def _list_difference(
        self, long_list: List[Any], short_list: List[Any]
    ) -> List[Any]:
        short_set = set(short_list)
        return [i for i in long_list if not i in short_set]

    def get_y_pred_train(self) -> LabelList:
        if not self.y_pred_train_calculated:
            self.y_pred_train = self.model.predict(self.local_X_train).tolist()
            self.y_pred_train_calculated = True
        return self.y_pred_train

    def get_y_pred_test(self) -> LabelList:
        if not self.y_pred_test_calculated:
            self.y_pred_test = self.model.predict(
                self.X[self.global_test_idx, :]
            ).tolist()
            self.y_pred_test_calculated = True
        return self.y_pred_test
