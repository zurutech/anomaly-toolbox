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

"""Anomaly detection dataset interface."""

import abc
from typing import Tuple, Union

import tensorflow as tf


class AnomalyDetectionDataset(abc.ABC):
    """
    Anomaly detection dataset interface.
    Note:
        The label 1 is for the ANOMALOUS class.
        The label 0 is for the NORMAL class.
    """

    def __init__(self):
        self._train = None
        self._train_normal = None
        self._train_anomalous = None

        self._test = None
        self._test_normal = None
        self._test_anomalous = None

        self._validation = None
        self._validation_normal = None
        self._validation_anomalous = None

        self._channels = 1

        self._anomalous_label = tf.constant(1)
        self._normal_label = tf.constant(0)

    @staticmethod
    def linear_conversion(
        image: tf.Tensor, new_min: tf.Tensor, new_max: tf.Tensor
    ) -> tf.Tensor:
        """Linearly convert image from it's actual value range,
        to a new range (new_min, new_max).
        Useful to change the values of an image in a new range.

        Args:
            image: float tensor
            new_min: float value or tensor
            new_max: new maximum value, float value or tensor
        Returns:
            a tensor with the same shape and type of image, but
            with values in  [new_min, new_max]
        """
        old_min = tf.reduce_min(image)
        old_max = tf.reduce_max(image)
        return (image - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

    @property
    def channels(self) -> int:
        """The last dimension of the elements in the dataset.
        e.g. 3 if the dataset is a dataset of RGB images or 1
        if they are grayscale."""
        return self._channels

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
    def validation(self) -> tf.data.Dataset:
        """The complete test dataset with both positive and negatives.
        The labels are always 2."""
        return self._validation

    @property
    def validation_normal(self) -> tf.data.Dataset:
        """Subset of the validation dataset: only positive."""
        return self._validation_normal

    @property
    def validation_anomalous(self) -> tf.data.Dataset:
        """Subset of the validation dataset: only negative."""
        return self._validation_anomalous

    @abc.abstractmethod
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
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains labels, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            class_label: If the raw dataset contains different classes (each one
                         containing both positive and negative samples) we can select
                         only one class to focus on (e.g. a dataset of industrial
                         defects on industrial objects composed of transistors and
                         pills and we are interested only in transistors and not on pills).
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use for
                          the processed images.
        """
        raise NotImplementedError

    @staticmethod
    def pipeline(
        dataset: tf.data.Dataset,
        new_size: Tuple[int, int],
        batch_size: int,
        cache: bool,
        shuffle_buffer_size: int,
        is_training: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ) -> tf.data.Dataset:
        """Given a dataset, configure it applying the chain of
        map, filter, shuffle and all the needed methods of the tf.data.Dataset.
        Args:
            dataset: the input dataset with elements (x,y), where x is an image
                     and y the scalar label.
            new_size: (H,W) of the output image. NEAREST_NEIGHBOR interpolation is used.
            batch_size: the dataset batch size
            cache: when true, calls the `.cache()` method on the dataset before
                   returning it.
            shuffle_buffer_size: buffer size used during the tf.data.Dataset.shuffle call.
            is_training: when true, shuffles the dataset element using a shuffle buffer with
                         size shuffle_buffer_size.
            drop_remainder: if True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: a Tuple (min, max) containing the output range to use
                          for the processed images.
        Returns:
            The configured dataset object.
        """
        dataset = dataset.map(
            lambda image, label: (
                tf.image.resize(
                    image, size=new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                ),
                label,
            )
        )

        dataset = dataset.map(
            lambda image, label: (tf.cast(image, tf.float32) / 255.0, label)
        )

        if output_range != (0.0, 1.0):
            dataset = dataset.map(
                lambda image, label: (
                    AnomalyDetectionDataset.linear_conversion(
                        image, output_range[0], output_range[1]
                    ),
                    label,
                )
            )

        if is_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if cache:
            dataset = dataset.cache()
        return dataset
