import tensorflow as tf


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
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        y_true=tf.ones_like(x) * label, y_pred=x
    )


def min_max(positive, negative, label_smooth: bool = False):
    """
    Return the discriminator (min max) loss.

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


def feature_matching_loss(feature_a, feature_b):
    """
    Return the feature matching loss between a and b.

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


def residual_loss(x, Gz) -> tf.Tensor:
    """
    Return the residual loss between x and Gz.

    Args:
        x: The original images batch, 4D
        Gz: The generated images

    Returns:
        sum | Gz - x |
    """
    assert x.shape == Gz.shape
    flat = (-1, x.shape[1] * x.shape[2] * x.shape[3])
    return tf.reduce_mean(tf.reshape(tf.abs(Gz - x), flat))
