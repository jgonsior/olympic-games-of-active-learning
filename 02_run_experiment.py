from abc import ABC, abstractmethod
import csv
import random
from typing import Any, List
import pandas as pd
from datasets import DATASET, load_dataset, split_dataset
from misc.config import Config
from misc.logging import log_it
from ressources.data_types import (
    AL_STRATEGY,
    LEARNER_MODEL,
    al_strategy_to_python_classes_mapping,
    learner_models_to_classes_mapping,
)
from sklearn.metrics import classification_report
import numpy as np


class AL_Experiment(ABC):
    config: Config

    def __init__(self) -> None:
        self.config = Config()

        log_it(
            f"Executing Job # {self.config.WORKER_INDEX} of workload {self.config.WORKLOAD_FILE_PATH}"
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
        dataset = DATASET(self.config.EXP_DATASET)
        df = load_dataset(dataset, self.config)

        np.random.seed(self.config.EXP_RANDOM_SEED)
        random.seed(self.config.EXP_RANDOM_SEED)

        # load dataset
        (
            self.X,
            self.Y,
            self.train_idx,
            self.test_idx,
            self.label_idx,
            self.unlabel_idx,
        ) = split_dataset(df, self.config)

        self.prepare_dataset()

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

        for iteration in range(0, total_amount_of_iterations):
            log_it(f"#{iteration}")

            # select some samples by indice to label
            select_ind = self.query_AL_strategy()
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

        # save metric results into a single file
        output_df = pd.json_normalize(confusion_matrices, sep="_")  # type: ignore
        output_df["selected_indices"] = selected_indices

        log_it(f"saving to {self.config.METRIC_RESULTS_FILE_PATHES}")
        output_df.to_csv(self.config.METRIC_RESULTS_FILE_PATHES, index=None)

        # save workload parameters in the workload_done_file
        workload = {}

        for k, v in Config.__annotations__.items():
            if (
                k.startswith("EXP_")
                and not str(v).startswith("typing.List[")
                and not k.startswith("EXP_GRID_")
            ):
                workload[k] = v

        print(workload)

        with open(self.config.DONE_WORKLOAD_PATH, "a") as f:
            w = csv.DictWriter(f, fieldnames=workload.keys())

            if self.config.DONE_WORKLOAD_PATH.stat().st_size == 0:
                print("write headers first")
                w.writeheader()
            w.writerow(workload)

    # efficient list difference
    def _list_difference(
        self, long_list: List[Any], short_list: List[Any]
    ) -> List[Any]:
        short_set = set(short_list)
        return [i for i in long_list if not i in short_set]


class ALIPY_AL_Experiment(AL_Experiment):
    def get_AL_strategy(self):
        al_strategy = AL_STRATEGY(self.config.EXP_STRATEGY)
        al_strategy = al_strategy_to_python_classes_mapping[al_strategy][0](
            X=self.X, y=self.Y, **al_strategy_to_python_classes_mapping[al_strategy][1]
        )
        self.al_strategy = al_strategy

    def query_AL_strategy(self) -> List[int]:
        return self.al_strategy.select(
            self.label_idx,
            self.unlabel_idx,
            model=self.model,
            batch_size=self.config.EXP_BATCH_SIZE,
        ).tolist()

    # dataset in numpy format and indice lists are fine as it is
    def prepare_dataset(self):
        pass


al_experiment = ALIPY_AL_Experiment()
al_experiment.run_experiment()
