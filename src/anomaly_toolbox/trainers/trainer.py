"""Trainer generic structure."""

from typing import Dict, List

import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class Trainer:
    """
    The Trainer represent the class to be used by all trainer objects.
    The class has the basic members that every trainer should have.
    """
    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
    ):
        """
        Trainer ctor.

        Args:
            dataset: The anomaly detection dataset.
            hps: all the hyperparameters needed.
            summary_writer: The tf.summary.SummaryWriter object to keep track of the training
            procedure.

        """
        self._dataset = dataset
        self._hps = hps
        self._summary_writer = summary_writer
        self._keras_metrics = []

    @property
    def keras_metrics(self) -> List[tf.keras.metrics.Metric]:
        """Return the list of keras metrics to measure during the training."""
        return self._keras_metrics

    def _reset_keras_metrics(self) -> None:
        """
        Reset all the metrics.
        """
        for metric in self.keras_metrics:
            metric.reset_states()
