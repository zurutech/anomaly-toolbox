from functools import partial
from typing import Tuple

import tensorflow as tf
from tensorflow._api.v2 import data
import tensorflow_datasets as tfds
from tensorflow.python.ops.image_ops_impl import ResizeMethod, resize_nearest_neighbor


class MNISTDataset:
    def __init__(self):
        (self.ds_train, self.ds_test), self.ds_info = tfds.load(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

    def assemble_datasets(
        self,
        anomalous_label: int,
        batch_size: int,
        new_size: Tuple[int, int] = (32, 32),
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        pipeline = partial(
            MNISTDataset.pipeline,
            size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
        )
        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: label == anomalous_label
        is_normal = lambda _, label: label != anomalous_label
        # --- Train Data ---
        ds_train_anomalous = self.ds_train.filter(is_anomalous).apply(pipeline_train)
        ds_train_normal = self.ds_train.filter(is_normal).apply(pipeline_train)
        # --- Test Data ---
        ds_test_anomalous = self.ds_test.filter(is_anomalous).apply(pipeline_test)
        ds_test_normal = self.ds_test.filter(is_normal).apply(pipeline_test)
        return ds_train_normal, ds_train_anomalous, ds_test_normal, ds_test_anomalous

    @staticmethod
    def scale(image, label):
        """Normalize images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    @staticmethod
    def resize(image, label, size):
        """Resize the data using Nearest Neighbor."""
        return (
            tf.image.resize(
                image,
                size=size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            ),
            label,
        )

    @staticmethod
    def pipeline(
        dataset: tf.data.Dataset,
        size: Tuple[int, int],
        batch_size: int,
        cache: bool,
        shuffle_buffer_size: int,
        is_training: bool = True,
    ) -> tf.data.Dataset:
        dataset = dataset.map(partial(MNISTDataset.resize, size=size))
        dataset = dataset.map(MNISTDataset.scale)
        if is_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        if cache:
            dataset = dataset.cache()
        return dataset
