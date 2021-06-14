"""Trainer generic structure."""

from typing import Dict, List

import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class Trainer:
    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
    ):
        """Initialize the Trainer."""
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
