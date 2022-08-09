from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import random
import time
from typing import TYPE_CHECKING, Any, List
import pandas as pd
from datasets import DATASET, load_dataset, split_dataset
from misc.logging import log_it

if TYPE_CHECKING:
    from misc.config import Config

from sklearn.metrics import classification_report
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning


class AL_Experiment(ABC):
    config: Config

    def __init__(self, config: Config) -> None:
        self.config = config

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
            self.label_idx,
            self.unlabel_idx,
        ) = split_dataset(dataset_tuple, self.config)

        self.prepare_dataset()

        from ressources.data_types import (
            learner_models_to_classes_mapping,
        )

        # load ml model
        model_instantiation_tuple = learner_models_to_classes_mapping[
            self.config.EXP_LEARNER_MODEL
        ]
        self.model = model_instantiation_tuple[0](**model_instantiation_tuple[1])

        # initially train model on initally labeled data
        self.model.fit(X=self.X[self.label_idx, :], y=self.Y[self.label_idx])  # type: ignore

        # select the AL strategy to use
        self.get_AL_strategy()

        # either we stop until all samples are labeled, or earlier
        if self.config.EXP_NUM_QUERIES == 0:
            total_amount_of_iterations = (
                int(len(self.unlabel_idx) / self.config.EXP_BATCH_SIZE) + 1
            )
        else:
            total_amount_of_iterations = self.config.EXP_NUM_QUERIES

        # the metrics we want to analyze later on
        confusion_matrices: List[np.ndarray] = []
        selected_indices: List[List[int]] = []

        log_it(f"Running for a total of {total_amount_of_iterations} iterations")

        start_time = time.process_time()

        for iteration in range(0, total_amount_of_iterations):
            if len(self.unlabel_idx) == 0:
                log_it("early stopping")
                break

            log_it(f"#{iteration}")

            # only use the query strategy if there are actualy samples left to label
            if len(self.unlabel_idx) > self.config.EXP_BATCH_SIZE:
                select_ind = self.query_AL_strategy()
            else:
                # if we have labeled everything except for a small batch -> return that
                select_ind = self.unlabel_idx
            self.label_idx = self.label_idx + select_ind

            self.unlabel_idx = self._list_difference(self.unlabel_idx, select_ind)

            # save indices for later
            selected_indices.append(select_ind)

            # update our learner model
            self.model.fit(X=self.X[self.label_idx, :], y=self.Y[self.label_idx])  # type: ignore

            # prediction on test set for metrics
            pred = self.model.predict(self.X[self.test_idx, :])  # type: ignore

            current_confusion_matrix = classification_report(
                y_true=self.Y[self.test_idx],
                y_pred=pred,
                output_dict=True,
                zero_division=0,
            )

            confusion_matrices.append(current_confusion_matrix)

        end_time = time.process_time()

        # save metric results into a single file
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices

        log_it(f"saving to {self.config.METRIC_RESULTS_FILE_PATH}")
        output_df.to_csv(self.config.METRIC_RESULTS_FILE_PATH, index=None)

        # save workload parameters in the workload_done_file
        workload = {}

        from misc.config import Config

        for k, v in Config.__annotations__.items():
            if (
                k.startswith("EXP_")
                and not str(v).startswith("typing.List[")
                and not k.startswith("EXP_GRID_")
            ):
                workload[k] = self.config.__getattribute__(k)
        workload["duration"] = end_time - start_time
        log_it(str(workload))

        with open(self.config.DONE_WORKLOAD_PATH, "a") as f:
            w = csv.DictWriter(f, fieldnames=workload.keys())

            if self.config.DONE_WORKLOAD_PATH.stat().st_size == 0:
                log_it("write headers first")
                w.writeheader()
            w.writerow(workload)

    # efficient list difference
    def _list_difference(
        self, long_list: List[Any], short_list: List[Any]
    ) -> List[Any]:
        short_set = set(short_list)
        return [i for i in long_list if not i in short_set]
