import abc
from typing import Tuple

import tensorflow as tf


class AnomalousDataset(abc.ABC):
    """Anomaly detection dataset interface."""

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
            - ds_train_normal: train dataset with normal data
            - ds_train_anomalous: train dataset with anomalous data
            - ds_test_normal: test dataset with normal data
            - ds_test_anomalous: test dataset with anomalous data
        """
