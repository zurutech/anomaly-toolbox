"""Several Implementation from AnoGAN paper."""

from typing import Any, List, Tuple

import tensorflow as tf
import tensorflow.keras as keras


class AnoGANAssembler:
    @staticmethod
    def assemble_generator(
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
        # Construct the zeroth block
        # TODO: Add l2 regularizer
        x = keras.layers.Dense(4 * 4 * filters * 2)(input_layer)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Reshape((-1, 4, 4, filters * 2))(x)

        # -----------
        # Construct the the first block
        x = keras.layers.Conv2DTranspose(
            filters,
            kernel_size=5,
            strides=2,
            padding="same",
        )(x)
        x = keras.layers.ReLU()(x)

        # -----------
        # Construct the various intermediate blocks
        vector_size = 8
        while vector_size < output_dimension[0] // 2:
            vector_size = vector_size * 2
            filters = filters // 2
            x = keras.layers.Conv2DTranspose(
                filters,
                kernel_size=5,
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
            kernel_size=5,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh",
        )(x)

        generator = tf.keras.Model(input_layer, x, name="anogan_generator")
        return generator

    @staticmethod
    def assemble_discriminator(
        input_dimension: Tuple[int, int, int],
        filters: int,
    ) -> tf.keras.Model:
        """
        GANomaly Encoder implementation as a :obj:`tf.keras.Model`.

        Args:
            filters of the first convolution (there will be log2(channel) conv)
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # -----------
        # Construct the the first block
        x = keras.layers.Conv2D(
            filters,
            kernel_size=5,
            strides=2,
            padding="same",
            use_bias=False,
        )(input_layer)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # -----------
        # Construct the various intermediate blocks
        channel_size = (
            input_dimension[0] // 2
        )  # Actually the side of the image aka c x c x filters
        while channel_size > 4:
            filters = filters * 2
            channel_size = channel_size // 2
            x = keras.layers.Conv2D(
                filters,
                kernel_size=5,
                strides=2,
                padding="same",
                use_bias=False,
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # -----------
        # Construct the final layer
        # x = keras.layers.Conv2D(
        #     latent_space_dimension,
        #     kernel_size=4,
        #     strides=1,
        #     padding="same",
        #     use_bias=False,
        # )(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)

        encoder = tf.keras.Model(input_layer, x, name="ganomaly_encoder")
        return encoder


class AnoGANMNISTAssembler:
    """Hardcoded GAN for the MNIST dataset."""

    @staticmethod
    def assemble_generator() -> keras.Model:
        input_layer = keras.layers.Input()

        x = keras.layers.Dense(7 * 7 * 128)(input_layer)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.Reshape((7, 7, 128))(x)

        x = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)
        x = keras.layers.Conv2D(64, (2, 2), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)
        x = keras.layers.Conv2D(64, (5, 5), padding="same")(x)
        x = keras.layers.Activation("tanh")(x)
        model = keras.Model(input_layer, x, name="anogan_mnist_generator")
        return model

    @staticmethod
    def assemble_discriminator() -> keras.Model:
        input_layer = keras.layers.Input(shape=(28, 28, 1))

        x = keras.layers.Conv2D(64, (5, 5), padding="same")(input_layer)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1)(x)

        model = keras.Model(input_layer, x, name="anogan_mninst_discriminator")
        return model
