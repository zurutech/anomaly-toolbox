"""GANomaly implementation."""

from os import name
from typing import Any, List, Tuple
from numpy.lib.function_base import vectorize

import tensorflow as tf
import tensorflow.keras as keras

# TODO: Add support for extra layers
class GANomalyAssembler:
    @staticmethod
    def assemble_encoder(
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 100,
    ) -> tf.keras.Model:
        """
        GANomaly Encoder implementation as a :obj:`tf.keras.Model`.

        Args:
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # -----------
        # Construct the the first block
        x = keras.layers.Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
        )(input_layer)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # -----------
        # Construct the various intermediate blocks
        channel_size = input_dimension[0] // 2
        while channel_size > 4:
            filters = filters * 2
            channel_size = channel_size // 2
            x = keras.layers.Conv2D(
                filters,
                kernel_size=4,
                strides=2,
                padding="same",
                use_bias=False,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # -----------
        # Construct the final layer
        x = keras.layers.Conv2D(
            latent_space_dimension,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
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
        GANomaly Encoder implementation as a :obj:`tf.keras.Model`.

        Args:
            input_dimension: Dimension of the Latent vector produced by the Encoder.
            output: Desired dimension of the output vector.
            filters: Filters of the first transposed convolution.
        """
        input_layer = keras.layers.Input(shape=(1, 1, input_dimension))

        # -----------
        # Construct the the first block
        x = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
        )(input_layer)
        x = keras.layers.ReLU()(x)

        # -----------
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
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)

        # -----------
        # Construct the final layer
        x = keras.layers.Conv2DTranspose(
            output_dimension[-1],
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh",
        )(x)

        decoder = tf.keras.Model(input_layer, x, name="ganomaly_decoder")
        return decoder


class GANomalyDiscriminator(keras.Model):
    def __init__(
        self,
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 1,
    ):
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

    def call(self, inputs, training, mask):
        features = self.features_extractor(inputs)
        classification = self.classifier(features)
        return classification, features


class GANomalyGenerator(keras.Model):
    def __init__(self, input_dimension, filters, latent_space_dimension):
        super().__init__()
        self.encoder_1 = GANomalyAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension
        )
        self.encoder_2 = GANomalyAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension
        )
        self.decoder = GANomalyAssembler.assemble_decoder(
            latent_space_dimension, input_dimension, filters
        )

    def call(self, inputs, training, mask):
        latent_i = self.encoder_1(inputs)
        generated_data = self.decoder(latent_i)
        latent_o = self.encoder_2(generated_data)
        return generated_data, latent_i, latent_o
