"""
BiGAN losses according to EGBAD.

The losses here defined are the standard GAN losses (non saturating + sum BCEs).
"""

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
    real_loss = keras.losses.binary_crossentropy(
        # from_logits=True
        tf.ones_like(d_real),
        d_real,
        from_logits=False,
    )
    generated_loss = keras.losses.binary_crossentropy(
        # from_logits=True
        tf.zeros_like(d_gen),
        d_gen,
        from_logits=False,
    )

    return tf.reduce_mean(real_loss + generated_loss)


def encoder_loss(d_real):
    return keras.losses.binary_crossentropy(
        # from_logits=True
        tf.zeros_like(d_real),
        d_real,
        from_logits=False,
    )


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
    output_loss = keras.losses.mean_squared_error(feature_a, feature_b)
    return output_loss


def residual_loss(x, Gz, axis=-1):
    """
    Return the residual loss between x and Gz.

    Args:
        x: The original images batch, 4D
        Gz: The generated images
        axis: The axis on which perform the reduce operation, deafault: -1

    Returns:
        sum | Gz - x |
    """
    assert x.shape == Gz.shape
    flat = (-1, x.shape[1] * x.shape[2] * x.shape[3])
    return tf.reduce_mean(tf.reshape(tf.abs(Gz - x), shape=flat), axis=axis)


# def min_max(positive, negative, label_smooth=False) -> tf.Tensor:
#     """
#     Return the discriminator (min max) loss.

#     Args:
#         positive: the discriminator output for the positive class: 2D tensor
#         negative: the discriminator output for the negative class: 2D tensor
#         smooth: when true, applies one-sided label smoothing

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
