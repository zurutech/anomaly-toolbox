"""Anomaly detection dataset interface."""

import abc
from typing import Optional, Tuple

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

        self._channels = 1

        self._anomalous_label = tf.constant(1)
        self._normal_label = tf.constant(0)

    @staticmethod
    def linear_conversion(image, new_min, new_max):
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
        new_min = tf.convert_to_tensor(new_min)
        new_max = tf.convert_to_tensor(new_max)
        return tf.clip_by_value(
            (((image - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min,
            clip_value_min=new_min,
            clip_value_max=new_max,
        )

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
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Optional[int] = None,
        shuffle_buffer_size: int = 10000,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: the dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: if the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            shuffle_buffer_size: buffer size used during the tf.data.Dataset.shuffle call.
            cache: if True, cache the dataset
            drop_remainder: if True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: a Tuple (min, max) containing the output range to use for
                          the processed images.
        """

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
            dataset: the input dataset with elmenents (x,y), where x is an image
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

        if output_range[0] != 0 and output_range[1] != 1:

            def squash_fn():
                """Closure. It returns a function(image,label), that applies
                the AnomalyDetectionDataset.linear_conversion on the image
                to squash the values in [output_range[0], output_range[1]].
                The function returns the pair (new_image, label).
                """

                def fn(image, label):
                    return (
                        AnomalyDetectionDataset.linear_conversion(
                            image, output_range[0], output_range[1]
                        ),
                        label,
                    )

                return fn

            dataset = dataset.map(squash_fn())

        if is_training:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        if cache:
            dataset = dataset.cache()
        return dataset
