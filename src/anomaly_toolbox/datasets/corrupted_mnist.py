"""Corrupoted MNIST dataset, splitted to be used for anomaly detection."""

import tensorflow_datasets as tfds

from .mnist import MNIST


class CorruptedMNIST(MNIST):
    """Corrupted MNIST dataset, splitted to be used for anomaly detection."""

    def __init__(self, corruption_type="shot_noise"):
        """Corrupted MNIST dataset, splitted to be used for anomaly detection.
        Args:
            corruption_type: one among the available corruptions as reported in
                             https://www.tensorflow.org/datasets/catalog/mnist_corrupted
        """
        super().__init__()
        (self._train, self._test), _ = tfds.load(
            f"mnist_corrupted/{corruption_type}",
            split=["train", "test"],
            as_supervised=True,
            with_info=True,
        )
