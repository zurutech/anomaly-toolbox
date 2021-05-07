"""
GANomaly Architecture implementation.

Due to some of the characteristics of the Architecture, this module is structured as such:

GANomalyAssembler -> Primitive Encoder and Decoder
GANomalyGenerator -> Built on top of the Decoder Primitive
GANomalyDiscriminator -> Built on top of the Encoder Primitive
"""

from typing import Any, Tuple

import tensorflow as tf
import tensorflow.keras as keras

__ALL__ = ["GANomalyAssembler", "GANomalyGenerator", "GANomalyDiscriminator"]

KERNEL_INITIALIZER = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# TODO: Add support for extra layers
class GANomalyAssembler:
    """Assembler providind GANomaly primitive Encoder and Decoder architectures."""

    @staticmethod
    def assemble_encoder(
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 100,
    ) -> tf.keras.Model:
        """
        Assemble the GANomaly Encoder as a :obj:`tf.keras.Model` using Keras Functional API.

        Note:
            This is designed to work with any image size ina fully-convolutional way.

        Args:
            input_dimension: Tuple[int, int, int] representing the shape of the input data.
            filters: Filters of the first convolution.
            latent_space_dimension: Dimension of the resulting encoded represention.

        Return:
            The assembled model.
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # Construct the the first block
        x = keras.layers.Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INITIALIZER,
        )(input_layer)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # Construct the various intermediate blocks
        channel_size = input_dimension[0] // 2
        while channel_size > 4:
            filters *= 2
            x = keras.layers.Conv2D(
                filters,
                kernel_size=4,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer=KERNEL_INITIALIZER,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)

            channel_size //= 2

        # Construct the final layer
        x = keras.layers.Conv2D(
            latent_space_dimension,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)

        encoder = tf.keras.Model(input_layer, x, name="ganomaly_encoder")
        return encoder

    @staticmethod
    def assemble_decoder(
        input_dimension: int,
        output_dimension: Tuple[int, int, int],
        filters: int,
    ) -> tf.keras.Model:
        """
        Assemble GANomaly Decoder as a :obj:`tf.keras.Model` using the Functional API.

        Args:
            input_dimension: Dimension of the Latent vector produced by the Encoder.
            output_dimension: Desired dimension of the output vector.
            filters: Filters of the first transposed convolution.

        Returns:
            The assembled model.
        """
        input_layer = keras.layers.Input(shape=(1, 1, input_dimension))

        # Construct the the first block
        x = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
            kernel_initializer=KERNEL_INITIALIZER,
        )(input_layer)
        x = keras.layers.ReLU()(x)

        # Construct the various intermediate blocks
        vector_size = 4
        while vector_size < output_dimension[0] // 2:
            vector_size = vector_size * 2
            filters = filters * 2
            x = keras.layers.Conv2DTranspose(
                filters,
                kernel_size=4,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer=KERNEL_INITIALIZER,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        # Construct the final layer
        x = keras.layers.Conv2DTranspose(
            output_dimension[-1],
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh",
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)

        decoder = tf.keras.Model(input_layer, x, name="ganomaly_decoder")
        return decoder


class GANomalyDiscriminator(keras.Model):
    """Implementation of the GANomaly Discriminator using :class:`GANomalyAssembler`."""

    def __init__(
        self,
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 1,
    ) -> None:
        """Initialize the the model."""
        super().__init__()
        layers = GANomalyAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension
        ).layers

        self.features_extractor = keras.Sequential(
            layers[:-1], name="ganomaly_discriminator_features_extractor"
        )
        self.classifier = keras.Sequential(
            [layers[-1], keras.layers.Flatten()],
            name="ganomaly_discriminator_classifier",
        )

    def call(self, inputs, training: bool = True) -> Tuple[Any, Any]:
        """Perform the forward pass."""
        features = self.features_extractor(inputs, training=training)
        classification = self.classifier(features, training=training)
        return classification, features


class GANomalyGenerator(keras.Model):
    """Implementation of the GANomaly Generator using :class:`GANomalyAssembler`."""

    def __init__(self, input_dimension, filters, latent_space_dimension):
        """Initialize the the model."""
        super().__init__()
        self.encoder1 = GANomalyAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension
        )

        self.decoder = GANomalyAssembler.assemble_decoder(
            latent_space_dimension, input_dimension, filters
        )

        self.encoder2 = GANomalyAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension
        )

    def call(
        self, inputs, training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Perform the forward pass."""
        latent_i = self.encoder1(inputs, training=training)
        generated_data = self.decoder(latent_i, training=training)
        latent_o = self.encoder2(generated_data)
        return latent_i, generated_data, latent_o
