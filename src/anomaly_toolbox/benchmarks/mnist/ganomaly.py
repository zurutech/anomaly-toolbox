from sys import path
from typing import Optional

import numpy as np
import tensorflow as tf

from anomaly_toolbox.benchmarks.ganomaly import compute_prc
from anomaly_toolbox.datasets import MNIST
from anomaly_toolbox.experiments import GANomalyExperimentMNIST
from anomaly_toolbox.predictors import GANomalyPredictor

__ALL__ = ["GANomalyMNISTBenchmark"]


class GANomalyMNISTBenchmark:
    def __init__(self, run_path):
        self.run_path = run_path
        self.predictor = GANomalyPredictor()

    def load_from_savedmodel(self):
        self.predictor.load_from_savedmodel(
            self.run_path + "/" + "generator",
            self.run_path + "/" + "discriminator",
        )
        return self

    def run(
        self, anomalous_label: Optional[int] = None, batch_size: Optional[int] = None
    ):
        # TODO: Fetch the values from the Experiment the user did not pass them
        if not anomalous_label:
            anomalous_label = 2
        if not batch_size:
            batch_size = 32

        ds_builder = MNIST()
        datasets = ds_builder.assemble_datasets(
            anomalous_label=anomalous_label, batch_size=batch_size, new_size=(32, 32)
        )
        anomaly_scores, labels = [], []
        for dataset in datasets[:-2]:
            _anomaly_scores, _labels = self.predictor.evaluate(dataset)
            anomaly_scores.extend(_anomaly_scores[0])
            labels.extend(_labels)

        # make labels binary
        best_ap = 0
        aucs = []

        for label in range(10):

            positive_mask = labels == anomalous_label
            negative_mask = labels != anomalous_label
            labels[positive_mask] = 1
            labels[negative_mask] = 0

            prc, _ = compute_prc(anomaly_scores, np.array(labels), f"{anomalous_label}")
            aucs.append(prc)

        print(
            f"Mean AUC: {np.mean(np.array(aucs))} || Best Average Precision: {best_ap}"
        )
