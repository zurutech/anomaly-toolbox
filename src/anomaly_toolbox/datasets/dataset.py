import abc
from typing import Tuple

import tensorflow as tf


class AnomalyDetectionDataset(abc.ABC):
    """
    Anomaly detection dataset interface.
    """

    def __init__(self):
        self._ds_train_normal = None
        self._ds_train_anomalous = None
        self._ds_test_normal = None
        self._ds_test_anomalous = None

    @property
    def train_normal(self):
        return self._ds_train_normal

    @property
    def train_anomalous(self):
        return self._ds_train_anomalous

    @property
    def test_normal(self):
        return self._ds_test_normal

    @property
    def test_anomalous(self):
        return self._ds_test_anomalous

    @property
    def dataset(self):
        return (
            self.train_normal,
            self.train_anomalous,
            self.test_normal,
            self.test_anomalous,
        )

    @abc.abstractmethod
    def assemble_datasets(
        self,
        anomalous_label: int,
        batch_size: int,
        new_size: Tuple[int, int] = (28, 28),
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Assemble train and test datasets.
        Returns:
            - train_normal: train-dataset with normal data.
            - train_anomalous: train-dataset with anomalous data.
            - test_normal: test-dataset with normal data.
            - test_anomalous: test-dataset with anomalous data.
        """
