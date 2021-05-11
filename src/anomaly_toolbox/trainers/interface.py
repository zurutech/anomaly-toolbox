from typing import Dict, List
import tensorflow as tf


class Trainer:
    _keras_metrics: List[tf.keras.metrics.Mean]
    ds_train: tf.data.Dataset
    ds_train_anomalous: tf.data.Dataset
    ds_test: tf.data.Dataset
    ds_test_anomalous: tf.data.Dataset

    def __init__(self, hps: Dict, summary_writer: tf.summary.SummaryWriter):
        """Initialize the Trainer."""
        self.hps = hps
        self.summary_writer = summary_writer

    def _reset_keras_metrics(self) -> None:
        for metric in self._keras_metrics:
            metric.reset_states()
