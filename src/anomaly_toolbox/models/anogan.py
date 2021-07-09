"""Implementation of the 28x28 input resolution models of AnoGAN."""
from typing import Tuple

import tensorflow as tf
import tensorflow.keras as k


class Generator(k.Sequential):
    """Generator in fully convolutional fashion.
    Input: 1x1x input_dimension.
    """

    def __init__(self, n_channels: int = 3, input_dimension: int = 128):
        """Generator model.
        Args:
            n_channels: Depth of the input image.
            input_dimension: The dimension of the latent vector.
        """
        super().__init__(
            [
                k.layers.InputLayer(input_shape=(input_dimension,)),
                k.layers.Dense(7 * 7 * 128, use_bias=False),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Reshape((7, 7, 128)),
                k.layers.Conv2DTranspose(
                    128, (2, 2), strides=(1, 1), use_bias=False, padding="same"
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    64, (2, 2), strides=(2, 2), use_bias=False, padding="same"
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    n_channels, (5, 5), strides=(2, 2), use_bias=False, padding="same"
                ),
                k.layers.Activation("tanh"),
            ]
        )


class Discriminator(k.Model):
    """Discriminator model. Expects a batch of 28x28x1 input images."""

    def __init__(self, n_channels: int = 3):
        """
        Args:
            n_channels: Depth of the input image.
        """
        super().__init__()

        self._features = k.Sequential(
            [
                k.layers.InputLayer(input_shape=(28, 28, n_channels)),
                k.layers.Conv2D(32, (5, 5), padding="same", strides=(2, 2)),
                k.layers.LeakyReLU(0.2),
                k.layers.Dropout(0.5),
            ]
        )

        self._classifier = k.Sequential(
            [
                k.layers.InputLayer(input_shape=(14, 14, 32)),
                k.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
                k.layers.LeakyReLU(0.2),
                k.layers.Dropout(0.5),
                k.layers.Flatten(),
                k.layers.Dense(1),
            ]
        )

    def call(
        self, inputs: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass.
        Args:
            inputs: Input batch.
            training: Toggle the model status from training to inference.
        Returns:
            out, features.

            out: The discriminator decision (single neuron, linear activation).
            features: The feature vector computed.
        """

        features = self._features(inputs, training=training)
        out = self._classifier(features, training=training)
        return out, features
