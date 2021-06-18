"""Anomaly detection dataset interface."""

import abc
from typing import Tuple

import tensorflow as tf


class AnomalyDetectionDataset(abc.ABC):
    """
    Anomaly detection dataset interface.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self):
        self._train_normal = None
        self._train_anomalous = None
        self._test_normal = None
        self._test_anomalous = None
        self._train = None
        self._test = None

        self._anomalous_label = tf.constant(1)
        self._normal_label = tf.constant(0)

    @property
    def anomalous_label(self) -> tf.Tensor:
        """Return the constant tensor used for anomalous label (1)."""
        return self._anomalous_label

    @property
    def normal_label(self) -> tf.Tensor:
        """Return the constant tensor used for normal data label (0)."""
        return self._normal_label

    @property
    def train_normal(self) -> tf.data.Dataset:
        """Subset of the training dataset: only positive."""
        return self._train_normal

    @property
    def train_anomalous(self) -> tf.data.Dataset:
        """Subset of the training dataset: only negative."""
        return self._train_anomalous

    @property
    def train(self) -> tf.data.Dataset:
        """The complete training dataset with both positive and negatives.
        The labels are always 2."""
        return self._train

    @property
    def test(self) -> tf.data.Dataset:
        """The complete test dataset with both positive and negatives.
        The labels are always 2."""
        return self._test

    @property
    def test_normal(self) -> tf.data.Dataset:
        """Subset of the test dataset: only positive."""
        return self._test_normal

    @property
    def test_anomalous(self) -> tf.data.Dataset:
        """Subset of the test dataset: only negative."""
        return self._test_anomalous

    @property
    def datasets(
        self,
    ) -> Tuple[tf.data.Dataset, ...]:
        """All the dataset in the following order:
        - train_normal
        - train_anomalous
        - test_normal
        - test_anomalous
        - train
        - test
        """
        return (
            self.train_normal,
            self.train_anomalous,
            self.test_normal,
            self.test_anomalous,
            self.train,
            self.test,
        )

    @abc.abstractmethod
    def configure(
        self,
        anomalous_label: int,
        batch_size: int,
        new_size: Tuple[int, int],
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None)."""
