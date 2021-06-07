from sys import path
from typing import Optional, List

import numpy as np
from numpy.lib.function_base import append
import tensorflow as tf
import tensorflow.keras as keras
from anomaly_toolbox.benchmarks.metrics import compute_prc
from anomaly_toolbox.datasets import MNIST
from anomaly_toolbox.experiments import GANomalyExperimentMNIST
from anomaly_toolbox.predictors import GANomalyPredictor

__ALL__ = ["GANomalyMNISTBenchmark"]


class GANomalyMNISTBenchmark:
    def __init__(self, run_path):
        self.run_path = run_path
        self.predictor = GANomalyPredictor()
        self.pr_auc_metric = keras.metrics.AUC(curve="PR")
        self.roc_auc_metric = keras.metrics.AUC()

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

        anomaly_scores: List[tf.Tensor] = []
        labels: List[tf.Tensor] = []
        for d in datasets:
            a, y = self.predictor.evaluate(d)
            anomaly_scores.append(a)
            labels.append(y)

        anomaly_scores: np.ndarray = tf.concat(anomaly_scores, axis=0).numpy()
        labels: np.ndarray = tf.concat(labels, axis=0).numpy()

        # Binarize labels
        positive_mask = labels == anomalous_label
        negative_mask = labels != anomalous_label
        labels[negative_mask] = 1
        labels[positive_mask] = 0

        self.pr_auc_metric.update_state(labels, anomaly_scores)
        self.roc_auc_metric.update_state(labels, anomaly_scores)
        print(f"Anomaly scores Keras-AUC-PR is: {self.pr_auc_metric.result().numpy()}")
        print(
            f"Anomaly scores Keras-AUC-ROC is: {self.roc_auc_metric.result().numpy()}"
        )

        # SciKitLearn computation
        pr_auc, average_precision = compute_prc(
            labels,
            anomaly_scores,
            file_name=self.run_path + "/" + "benchmark_mnist_pr_auc.png",
            plot=True,
        )
        print(f"Anomaly Scores ScikitLearn AUC-PR is {pr_auc}")
        print(f"Anomaly Scores ScikitLearn Average Precision is {average_precision}")
