"""MNIST dataset, splitted to be used for anomaly detection."""

from functools import partial
from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from .dataset import AnomalyDetectionDataset


class MNIST(AnomalyDetectionDataset):
    """MNIST dataset, split to be used for anomaly detection.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self):
        super().__init__()
        (self._train_raw, self._test_raw), _ = tfds.load(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

    def configure(
        self,
        anomalous_label: int,
        batch_size: int,
        new_size: Tuple[int, int] = (28, 28),
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
    ):
        pipeline = partial(
            self.pipeline,
            size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
        )

        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        # Train-data
        self._train_anomalous = (
            self._train_raw.filter(is_anomalous)
            .map(lambda x, y: (x, self.anomalous_label))
            .apply(pipeline_train)
        )
        self._train_normal = (
            self._train_raw.filter(is_normal)
            .map(lambda x, y: (x, self.normal_label))
            .apply(pipeline_train)
        )
        self._train = self._train_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.int32),
            )
        ).apply(pipeline_train)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, y: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, y: (x, self.normal_label))
            .apply(pipeline_test)
        )

        # Complete dataset with positive and negatives
        def _to_binary(x, y):
            if tf.equal(y, anomalous_label):
                return (x, self.anomalous_label)
            return (x, self.normal_label)

        self._test = self._test_raw.map(_to_binary).apply(pipeline_test)

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
