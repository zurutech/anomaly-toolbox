"""Collection of various loss functions."""

# TODO: Add residual loss
# TODO: Refactor here all common type of loss

import tensorflow as tf
from tensorflow import keras

__ALL__ = ["adversarial_loss", "feature_matching_loss"]


def adversarial_loss(d_real, d_gen):
    """
    Compute the adversarial (min-max) loss.

    Args:
        d_real: Output of the Discriminator when fed with real data.
        d_gen: Output of the Discriminator when fed with generated data.

    Returns:
        Loss on real data + Loss on generated data.
    """
    real_loss = keras.losses.binary_crossentropy(
        tf.ones_like(d_real), d_real, from_logits=True
    )
    generated_loss = keras.losses.binary_crossentropy(
        tf.zeros_like(d_gen), d_gen, from_logits=True
    )
    return real_loss + generated_loss


def feature_matching_loss(feature_a, feature_b):
    """
    Compute the adversarial feature matching loss.

    Args:
        feature_a: input image feature
        feature_b: generated image feature

    Returns:
        The value of the adversarial loss

    """
    return keras.losses.mean_squared_error(feature_a, feature_b)
