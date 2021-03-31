from typing import Callable
import tensorflow as tf
from tensorflow import keras


def discriminator_loss(d_real, d_gen):
    """
    Compute the discriminator loss.

    Args:
        d_real: Output of the Discriminator when fed with real data.
        d_gen: Output of the Discriminator when fed with generated data.

    Returns:
        Loss on real data + Loss on generated data.
    """

    print("d_real.shape: ", d_real, " || d_gen.shape: ", d_gen)
    real_loss = tf.nn.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(d_real), logits=d_real
    )
    generated_loss = tf.nn.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(d_gen), logits=d_gen
    )
    return real_loss + generated_loss


def adversarial_loss_bce(discriminator: keras.Model, generated_data):
    d_generated_data, _ = discriminator(generated_data)
    return tf.nn.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(d_generated_data),
        logits=d_generated_data,
    )


def adversarial_loss_fm(
    # discriminator: k.Model,
    discriminator: Callable,
    query_data,
    generated_data,
):
    """
    Compute the adversarial feature matching loss.

    Args:
        discriminator: The discriminator
        query_data: the input x data
        generated_data: the data generated with generator

    Returns:
        the value of the adversarial loss

    """
    _, d_query_feature = discriminator(query_data)
    _, d_generated_feature = discriminator(generated_data)
    # d_query = discriminator.intermediate(query_data)
    # d_generated = discriminator.intermediate(generated_data)
    print(
        "discriminator.intermediate(query_data): ",
        d_query_feature.shape,
        " | discriminator.intermediate(generated_data): ",
        d_generated_feature.shape,
    )
    return tf.losses.mean_squared_error(d_query_feature, d_generated_feature)


def contextual_loss(query_data: tf.Tensor, generated_data: tf.Tensor):
    """
    Compute the contextual loss.

    Args:
        query_data: The input x
        generated_data: The generated data x'

    Returns:
        The L1 loss

    """
    return keras.losses.MeanAbsoluteError()(query_data, generated_data)


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


# TODO: Test exchanging it with keras.binary_cross_entropy while using a sigmoid activation on the last layer
def bce(x, label):
    """
    Return the discrete binary cross entropy between x and the discrete label.

    Args:
        x: a 2D tensor
        label: the discrite label, aka, the distribution to match

    Returns:
        The binary cros entropy
    """
    assert len(x.shape) == 2 and len(label.shape) == 0
    return tf.nn.sigmoid_cross_entropy(tf.ones_like(x) * label, x)


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


def min_max(positive, negative, label_smooth: bool = False):
    """
    Return the discriminator (min max) loss).

    Args:
        positive: the discriminator output for the positive class: 2D tensor
        negative: the discriminator output for the negative class: 2D tensor
        smooth: when true, appiles one-sided label smoothing
    Returns:
        The sum of 2 BCE
    """
    if label_smooth:
        one = smooth(1.0)
    else:
        one = tf.constant(1.0)
    zero = tf.constant(0.0)
    d_loss = bce(positive, one) + bce(negative, zero)
    return d_loss


def feature_matching_loss(feature_a, feature_b) -> tf.Tensor:
    """Returns the feature matching loss between a and b
    Args:
        feature_a: first tensor: 2D tensor (bs, feature)
        feature_b: second tensor: 2D tensor (bs, feature)
    Returns:
        The feature matching between feature_a and feature_b
        mean|mean(feature_a) - mean(feature_b)|²
    """
    assert len(feature_a.shape) == 2 and len(feature_b.shape) == 2
    mean_a = tf.reduce_mean(feature_a, axis=0)
    mean_b = tf.reduce_mean(feature_b, axis=0)
    fm = tf.reduce_mean(tf.math.squared_difference(mean_a, mean_b))
    return fm
