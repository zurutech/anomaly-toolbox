"""GANomaly implementation."""

from typing import Any, List, Tuple

import tensorflow as tf
import tensorflow.keras as keras


class GANomaly:
    @staticmethod
    def assemble_block(
        x,
        filters,
        kernel_size: Tuple[int, int] = (4, 4),
        strides: Tuple[int, int] = (2, 2),
        padding="same",
        use_bias: bool = False,
        leaky_relu_alpha: float = 0.2,
        is_initial_block: bool = False,
        is_final_block: bool = False,
    ):

        convolution = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )(x)
        activation = keras.layers.LeakyReLU(leaky_relu_alpha)
        if is_initial_block:
            return activation(convolution)
        if is_final_block:
            return convolution
        normalization = keras.layers.BatchNormalization()
        return activation(normalization(convolution))

    @staticmethod
    def assemble_transpose_block(
        x,
        filters,
        kernel_size: Tuple[int, int] = (4, 4),
        strides: Tuple[int, int] = (2, 2),
        padding="same",
        use_bias: bool = False,
        is_initial_block: bool = False,
        is_final_block: bool = False,
    ):

        convolution = keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
        )(x)
        activation = keras.activations.tanh if is_final_block else keras.layers.ReLU()
        if is_initial_block:
            return activation(convolution)
        if is_final_block:
            return activation(convolution)
        normalization = keras.layers.BatchNormalization()
        return activation(normalization(convolution))

    @staticmethod
    def discriminator(
        input_dimension: Tuple,
        filters: List[int] = [64, 128, 256, 1],
        intermediate_layer_id: int = 2,
    ) -> Tuple[tf.keras.Model, Any]:
        """
        GANomaly Discriminator implementation as a :obj:`tf.keras.Model`.

        Args:
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # Construct the various blocks
        intermediate_layer = None
        x = None
        for i, f in enumerate(filters):
            # ----------
            # Initial Block
            if i == 0:
                x = GANomaly.assemble_block(
                    input_layer, filters=f, is_initial_block=True
                )
            # ----------
            # All the non-initial, non-final blocks
            elif i != len(filters) + 1:
                x = GANomaly.assemble_block(x, filters=f)

                # Intermediate Layer extracted when we are instantiating the intermediate_layer_id
                if i == intermediate_layer_id:
                    intermediate_layer = x
            # ----------
            # The final block
            else:
                x = GANomaly.assemble_block(x, filters=f, is_final_block=True)

        discriminator = tf.keras.Model(input_layer, x, name="ganomaly_discriminator")
        discriminator.summary()
        return discriminator, intermediate_layer

    @staticmethod
    def encoder(
        input_dimension: Tuple,
        filters: List[int] = [64, 128, 256],
        latent_space_dimension: int = 100,
    ) -> tf.keras.Model:
        """
        GANomaly Encoder implementation as a :obj:`tf.keras.Model`.

        Args:
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # Construct the various blocks
        x = None
        for i, f in enumerate(filters):
            # ----------
            # Initial Block
            if i == 0:
                x = GANomaly.assemble_block(
                    input_layer, filters=f, is_initial_block=True
                )
            # ----------
            # All the non-initial, non-final blocks
            else:
                x = GANomaly.assemble_block(x, filters=f)

        # ----------
        # The final block
        x = GANomaly.assemble_block(
            x, filters=latent_space_dimension, is_final_block=True
        )

        encoder = tf.keras.Model(input_layer, x, name="ganomaly_encoder")
        encoder.summary()
        return encoder

    @staticmethod
    def decoder(
        input_dimension: int,
        filters: List[int] = [256, 128, 64],
        output_depth: int = 3,
    ) -> tf.keras.Model:
        """
        GANomaly Encoder implementation as a :obj:`tf.keras.Model`.

        Args:
        """
        input_layer = keras.layers.Input(shape=input_dimension)

        # Construct the various blocks
        x = None
        for i, f in enumerate(filters):
            # ----------
            # Initial Block
            if i == 0:
                x = GANomaly.assemble_transpose_block(
                    input_layer,
                    filters=f,
                    is_initial_block=True,
                    strides=(1, 1),
                    padding="valid",
                )
            # ----------
            # All the non-initial, non-final blocks
            else:
                x = GANomaly.assemble_transpose_block(x, filters=f)

        # ----------
        # The final block
        x = GANomaly.assemble_block(
            x,
            filters=output_depth,
            is_final_block=True,
            padding="valid",
            strides=(1, 1),
        )

        encoder = tf.keras.Model(input_layer, x, name="ganomaly_encoder")
        encoder.summary()
        return encoder
