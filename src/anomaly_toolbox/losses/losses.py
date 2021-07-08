"""Collection of various loss functions."""

import tensorflow as tf
from tensorflow import keras as k


def adversarial_loss(d_real, d_gen):
    """
    Compute the adversarial (min-max) loss.

    Args:
        d_real: Output of the Discriminator when fed with real data.
        d_gen: Output of the Discriminator when fed with generated data.

    Returns:
        Loss on real data + Loss on generated data.
    """
    real_loss = k.losses.binary_crossentropy(
        tf.ones_like(d_real), d_real, from_logits=True
    )
    generated_loss = k.losses.binary_crossentropy(
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
    return tf.reduce_mean(tf.math.squared_difference(feature_a, feature_b))


def residual_image(x, g_z):
    return tf.math.abs(x - g_z)


def residual_loss(x, g_z):
    return tf.reduce_mean(residual_image(x, g_z))
