from typing import Callable
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


def discriminator_loss(d_real, d_gen):
    """
    Compute the discriminator loss.

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


def adversarial_loss_bce(d_gen):
    return keras.losses.binary_crossentropy(
        tf.ones_like(d_gen), d_gen, from_logits=True
    )


def adversarial_loss_fm(feature_a, feature_b):
    """
    Compute the adversarial feature matching loss.

    Args:
        feature_a: input image feature
        feature_b: generated image feature

    Returns:
        the value of the adversarial loss

    """
    return keras.losses.mean_squared_error(feature_a, feature_b)


def contextual_loss(query_data: tf.Tensor, generated_data: tf.Tensor):
    """
    Compute the contextual loss.

    Args:
        query_data: The input x
        generated_data: The generated data x'

    Returns:
        The L1 loss

    """
    return keras.losses.mean_absolute_error(query_data, generated_data)


def encoder_loss(query_encoded: tf.Tensor, generated_encoded: tf.Tensor):
    """
    Compute the encoder loss.

    Args:
        query_encoded: It is the output of the first encoder (g_encoder)
        generated_encoded: It is the output of the second encoder (encoder)

    Returns:
        The L2 loss

    """
    return tf.losses.mean_squared_error(query_encoded, generated_encoded)


def smooth(label):
    """
    Given label, a float tensor, returns a random value in it's neighborhood.

    e.g: label = 0 -> value in [0, 0.3]
         label = 1 -> value in [0.7, 1.3]
    Returned values will always be positive

    Args:
        label: The input value

    Returns:
        The smoothed value
    """
    return tf.clip_by_value(
        tf.cast(label, tf.float32) + tf.random.normal(tf.shape(label), 0.3),
        clip_value_min=0.0,
        clip_value_max=label + 0.3,
    )


# def min_max(positive, negative, label_smooth: bool = False):
#     """
#     Return the discriminator (min max) loss).

#     Args:
#         positive: the discriminator output for the positive class: 2D tensor
#         negative: the discriminator output for the negative class: 2D tensor
#         smooth: when true, appiles one-sided label smoothing
#     Returns:
#         The sum of 2 BCE
#     """
#     if label_smooth:
#         one = smooth(1.0)
#     else:
#         one = tf.constant(1.0)
#     zero = tf.constant(0.0)
#     d_loss = bce(positive, one) + bce(negative, zero)
#     return d_loss
