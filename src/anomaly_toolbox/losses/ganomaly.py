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
