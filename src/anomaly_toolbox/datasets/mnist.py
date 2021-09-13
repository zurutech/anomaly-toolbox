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

"""MNIST dataset, split to be used for anomaly detection."""

from functools import partial
from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow_datasets as tfds

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class MNIST(AnomalyDetectionDataset):
    """MNIST dataset, split to be used for anomaly detection.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self):
        super().__init__()
        (self._train_raw, self._test_raw), info = tfds.load(
            "mnist",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )

        self._num_classes = info.features["label"].num_classes

    def configure(
        self,
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Union[int, str, None] = None,
        class_label: Union[int, str, None] = None,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size.
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            class_label: If the raw dataset contains different classes (each one
                         containing both positive and negative samples) we can select
                         only one class to focus on (e.g. a dataset of industrial
                         defects on industrial objects composed of transistors and
                         pills and we are interested only in transistors and not on pills).
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset.
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use
                          for the processed images.
        """

        pipeline = partial(
            self.pipeline,
            new_size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
            output_range=output_range,
        )

        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        # 60000 train images -> 6000 per class -> 600 per class in validation set
        # do not overlap wih train images -> 6000 - 600 per class in training set
        per_class_dataset = [
            self._train_raw.filter(lambda _, y: tf.equal(y, label))
            for label in tf.range(self._num_classes, dtype=tf.int64)
        ]

        validation_raw = per_class_dataset[0].take(600)
        train_raw = per_class_dataset[0].skip(600)
        for i in range(1, self._num_classes):
            validation_raw = validation_raw.concatenate(per_class_dataset[i].take(600))
            train_raw = train_raw.concatenate(per_class_dataset[i].skip(600))

        # Train-data
        self._train_anomalous = (
            train_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_train)
        )
        self._train_normal = (
            train_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_train)
        )
        self._train = train_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.int32),
            )
        ).apply(pipeline_train)

        # Validation data
        self._validation_anomalous = (
            validation_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._validation_normal = (
            validation_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._validation = validation_raw.map(
            lambda x, label: (
                x,
                tf.cast(tf.equal(label, anomalous_label), tf.int32),
            )
        ).apply(pipeline_test)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )

        # Complete dataset with positive and negatives
        def _to_binary(x, y):
            if tf.equal(y, anomalous_label):
                return (x, self.anomalous_label)
            return (x, self.normal_label)

        self._test = self._test_raw.map(_to_binary).apply(pipeline_test)
