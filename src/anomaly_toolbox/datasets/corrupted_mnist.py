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

"""Corrupted MNIST dataset, split to be used for anomaly detection."""

import tensorflow_datasets as tfds

from .mnist import MNIST


class CorruptedMNIST(MNIST):
    """Corrupted MNIST dataset, splitted to be used for anomaly detection."""

    def __init__(self, corruption_type="shot_noise"):
        """Corrupted MNIST dataset, split to be used for anomaly detection.
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
