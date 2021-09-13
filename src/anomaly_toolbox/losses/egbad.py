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

"""
BiGAN losses according to EGBAD.

The losses here defined are the standard GAN losses (non saturating + sum BCEs).
"""

import tensorflow as tf
from tensorflow import keras


class AdversarialLoss(keras.losses.Loss):
    """
    Adversarial loss.
    """

    def __init__(self, from_logits: bool = True):
        super().__init__()
        self._bce = keras.losses.BinaryCrossentropy(from_logits=from_logits)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        d_real = y_true
        d_gen = y_pred
        real_loss = self._bce(tf.ones_like(d_real), d_real)
        generated_loss = self._bce(tf.zeros_like(d_gen), d_gen)
        return real_loss + generated_loss


def residual_loss(x: tf.Tensor, g_z: tf.Tensor, axis: int = -1):
    """
    Return the residual loss between x and Gz.

    Args:
        x: The original images batch, 4D.
        g_z: The generated images.
        axis: The axis on which perform the reduce operation, default: -1.

    Returns:
        sum | Gz - x |
    """
    assert x.shape == g_z.shape
    flat = (-1, x.shape[1] * x.shape[2] * x.shape[3])
    return tf.reduce_mean(tf.reshape(tf.abs(g_z - x), shape=flat), axis=axis)


def generator_bce(d_gz: tf.Tensor, from_logits: bool = True):
    """
    Calculated the binary cross entropy loss of the generator.

    Args:
        d_gz: Discriminator classification on the generator reconstructed data.
        from_logits: True if the values are unbounded, False if they are a probability
        distribution [0, 1].

    Returns:
        The result of the keras.losses.BinaryCrossentropy function.
    """
    return keras.losses.BinaryCrossentropy(from_logits=from_logits)(
        tf.ones_like(d_gz), d_gz
    )


def encoder_bce(d_x: tf.Tensor, from_logits: bool = True):
    """
    Calculated the binary cross entropy loss of the encoder.

    Args:
        d_x: Discriminator classification on the real data.
        from_logits: True if the values are unbounded, False if they are a probability
        distribution [0, 1].

    Returns:
        The result of the keras.losses.BinaryCrossentropy function.
    """
    return keras.losses.BinaryCrossentropy(from_logits=from_logits)(
        tf.zeros_like(d_x), d_x
    )
