"""BiGAN Architecture Implementation as used in EGBAD."""

from typing import Tuple

import tensorflow as tf
import tensorflow.keras as keras

__ALL__ = ["EGBADBiGANAssembler"]


KERNEL_INITIALIZER = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = keras.initializers.RandomNormal(mean=1.0, stddev=0.02)


class EGBADBiGANAssembler:
    """Assembler providind BiGAN primitive Encoder and Decoder architectures."""

    @staticmethod
    def assemble_encoder(
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 100,
        l2_penalty: float = 0.0,
    ) -> tf.keras.Model:
        """
        Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model` using Keras Functional API.

        Note:
            This is designed to work with any image size ina fully-convolutional way.

        Args:
            input_dimension: Tuple[int, int, int] representing the shape of the input data.
            filters: Filters of the first convolution.
            latent_space_dimension: Dimension (along the depth) of the resulting encoded represention.
            l2_penalty: l2 regularization strenght

        Return:
            The assembled model.
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
            kernel_regularizer=keras.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
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
                kernel_regularizer=keras.regularizers.l2(l2_penalty),
                kernel_initializer=KERNEL_INITIALIZER,
            )(x)
            x = keras.layers.BatchNormalization(
                beta_initializer=ALMOST_ONE,
                gamma_initializer=ALMOST_ONE,
                momentum=0.1,
                epsilon=1e-5,
            )(x)
            x = keras.layers.LeakyReLU(alpha=0.2)(x)

        # -----------
        # Construct the final layer
        x = keras.layers.Conv2D(
            latent_space_dimension,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)

        encoder = tf.keras.Model(input_layer, x, name="bigan_encoder")
        return encoder

    @staticmethod
    def assemble_decoder(
        input_dimension: int,
        output_dimension: Tuple[int, int, int],
        filters: int,
        l2_penalty: float = 0.0,
    ) -> tf.keras.Model:
        """
        Assemble EGBAD-BiGAN Decoder as a :obj:`tf.keras.Model` using the Functional API.

        Args:
            input_dimension: Dimension of the Latent vector produced by the Encoder.
            output_dimension: Desired dimension of the output vector.
            filters: Filters of the first transposed convolution.

        Returns:
            The assembled model.
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
            kernel_regularizer=keras.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
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
                kernel_regularizer=keras.regularizers.l2(l2_penalty),
                kernel_initializer=KERNEL_INITIALIZER,
            )(x)
            x = keras.layers.BatchNormalization(
                beta_initializer=ALMOST_ONE,
                gamma_initializer=ALMOST_ONE,
                momentum=0.1,
                epsilon=1e-5,
            )(x)
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
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)

        decoder = tf.keras.Model(input_layer, x, name="bigan_decoder")
        return decoder

    @staticmethod
    def assemble_discriminator(
        input_dimension: Tuple[int, int, int],
        filters: int,
        latent_space_dimension: int = 100,
        l2_penalty: float = 0.0,
    ) -> keras.Model:
        encoder = EGBADBiGANAssembler.assemble_encoder(
            input_dimension, filters, latent_space_dimension, l2_penalty
        )
        input_layer = keras.layers.Input(shape=input_dimension)

        input_encoding = keras.layers.Input(shape=latent_space_dimension)
        d_z = keras.layers.Conv2D(
            128,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(input_encoding)
        d_z = keras.layers.LeakyReLU(0.2)(d_z)  # D(z) <-> 1x1x128

        concat_input = keras.layers.concatenate(
            [encoder(input_layer), d_z]
        )  # D(x|z) <-> 1x1x256

        fc = keras.layers.Conv2D(
            1024,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(concat_input)
        feature = keras.layers.LeakyReLU(0.2, name="feature")(fc)  # 1x1x1024

        out = keras.layers.Conv2D(
            1,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(
            feature
        )  # 1x1x1

        discriminator = keras.Model(
            inputs=[input_layer, input_encoding],
            outputs=[out, feature],
            name="bigan_discriminator",
        )
        return discriminator
