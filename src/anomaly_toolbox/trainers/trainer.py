# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer generic structure."""

from abc import abstractmethod
from pathlib import Path
from typing import Dict, Set

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
        log_dir: Path,
    ):
        """
        Trainer ctor.

        Args:
            dataset: The anomaly detection dataset.
            hps: instance of hyperparameters. Configure the trainer with this set of values.
                 NOTE: the hyperparameter supported and needed by the training procedure are
                 available in the `hyperparameters` property.
            summary_writer: The tf.summary.SummaryWriter object to keep track of the training
                            procedure.
            log_dir: the directory to use when saving something without using the summary writer.
        """
        self._log_dir = log_dir
        self._dataset = dataset
        self._hps = hps
        self._summary_writer = summary_writer
        self._keras_metrics = {}

    @property
    def keras_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        """Return the dictionary of keras metrics to measure during the training."""
        return self._keras_metrics

    @keras_metrics.setter
    def keras_metrics(self, metrics: Dict[str, tf.keras.metrics.Metric]):
        self._keras_metrics = metrics

    @staticmethod
    @abstractmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        raise NotImplementedError

    def _reset_keras_metrics(self) -> None:
        """
        Reset all the metrics.
        """
        for metric in self._keras_metrics.values():
            metric.reset_states()
