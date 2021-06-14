"""MNIST dataset, splitted to be used for anomaly detection."""

from functools import partial
from typing import Tuple, Optional

import tensorflow as tf
import tensorflow_datasets as tfds

from .dataset import AnomalyDetectionDataset


class MNIST(AnomalyDetectionDataset):
    """MNIST dataset, split to be used for anomaly detection."""

    def __init__(self):
        super().__init__()
        (self._ds_train, self._ds_test), self.ds_info = tfds.load(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

    def assemble_datasets(
        self,
        anomalous_label: int,
        batch_size: int,
        new_size: Tuple[int, int] = (28, 28),
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

        pipeline = partial(
            MNIST.pipeline,
            size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
        )

        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: label == anomalous_label
        is_normal = lambda _, label: label != anomalous_label

        # Train-data
        self._ds_train_anomalous = self._ds_train.filter(is_anomalous).apply(
            pipeline_train
        )
        self._ds_train_normal = self._ds_train.filter(is_normal).apply(pipeline_train)

        # Test-data
        self._ds_test_anomalous = self._ds_test.filter(is_anomalous).apply(
            pipeline_test
        )
        self._ds_test_normal = self._ds_test.filter(is_normal).apply(pipeline_test)

        return (
            self.train_normal,
            self.train_anomalous,
            self.test_normal,
            self.test_anomalous,
        )

    @staticmethod
    def pipeline(
        dataset: tf.data.Dataset,
        size: Tuple[int, int],
        batch_size: int,
        cache: bool,
        shuffle_buffer_size: int,
        is_training: bool = True,
        drop_remainder: bool = True,
    ) -> tf.data.Dataset:
        """Given a dataset, configure it applying the chain of
        map, filter, shuffle and all the needed methods of the tf.data.Dataset.
        """
        dataset = dataset.map(
            lambda image, label: (
                tf.image.resize(
                    image, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                ),
                label,
            )
        )
        dataset = dataset.map(
            lambda image, label: (tf.cast(image, tf.float32) / 255.0, label)
        )
        if is_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if cache:
            dataset = dataset.cache()
        return dataset
