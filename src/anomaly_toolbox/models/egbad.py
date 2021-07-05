"""BiGAN Architecture Implementation as used in EGBAD."""

import tensorflow as tf
import tensorflow.keras as k

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


class Encoder(k.Model):
    """
    Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 200):
        """
        Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model`.

        Note:
            This is designed to work with any image size in a fully-convolutional way.

        Args:
            n_channels: depth of the input image
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
        Return:
            The assembled model.
        """
        super().__init__()

        l2_penalty = 0.0
        filters = 32

        input_layer = k.layers.Input(shape=(28, 28, n_channels))

        x = k.layers.Conv2D(
            32,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
        )(input_layer)
        x = k.layers.LeakyReLU(alpha=0.2)(x)

        x = k.layers.Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)
        x = k.layers.LeakyReLU(alpha=0.2)(x)
        x = k.layers.BatchNormalization()(x)

        x = k.layers.Conv2D(
            128,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)
        x = k.layers.LeakyReLU(alpha=0.2)(x)
        x = k.layers.BatchNormalization()(x)

        x = k.layers.Flatten()(x)

        x = k.layers.Dense(
            units=latent_space_dimension, kernel_initializer=KERNEL_INITIALIZER
        )(x)

        self._encoder = k.Model(inputs=input_layer, outputs=x, name="bigan_encoder")

    def call(self, inputs, training=False):
        return self._encoder(inputs, training=training)


class Decoder(k.Model):
    """
    Assemble the EGBAD BiGAN Decoder as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 200):
        """
        Assemble the EGBAD BiGAN Decoder as a :obj:`tf.keras.Model`.

        Note:
            This is designed to work with any image size in a fully-convolutional way.

        Args:
            n_channels: depth of the output image.
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
        Return:
            The assembled model.
        """
        super().__init__()

        # output dimension is 28, 28
        l2_penalty = 0.0

        input_layer = k.layers.Input(shape=(latent_space_dimension,))

        x = k.layers.Dense(1024, activation="relu")(input_layer)
        x = k.layers.BatchNormalization(
            beta_initializer=ALMOST_ONE,
            gamma_initializer=ALMOST_ONE,
            momentum=0.1,
            epsilon=1e-5,
        )(x)
        x = k.layers.Dense(7 * 7 * 128, activation="relu")(x)
        x = k.layers.BatchNormalization(
            beta_initializer=ALMOST_ONE,
            gamma_initializer=ALMOST_ONE,
            momentum=0.1,
            epsilon=1e-5,
        )(x)

        x = tf.reshape(x, [-1, 7, 7, 128])

        x = k.layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
            activation="relu",
        )(x)
        x = k.layers.BatchNormalization(
            beta_initializer=ALMOST_ONE,
            gamma_initializer=ALMOST_ONE,
            momentum=0.1,
            epsilon=1e-5,
        )(x)
        x = k.layers.Conv2DTranspose(
            n_channels,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
            activation="tanh",
        )(x)

        self._decoder = k.Model(inputs=input_layer, outputs=x, name="bigan_decoder")

    def call(self, inputs, training=False):
        return self._decoder(inputs, training=training)


class Discriminator(k.Model):
    """
    Assemble the EGBAD BiGAN Discriminator as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 200):
        """
        Assemble the EGBAD BiGAN Discriminator as a :obj:`tf.keras.Model`.

        Note:
            This is designed to work with any image size in a fully-convolutional way.

        Args:
            n_channels: depth of the output image.
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
        Return:
            The assembled model.
        """
        super().__init__()
        input_dimension = (28, 28, n_channels)

        encoder = Encoder(n_channels, latent_space_dimension)

        # Input layers
        input_layer = k.layers.Input(shape=input_dimension)
        input_encoded_layer = k.layers.Input(shape=(latent_space_dimension,))

        # D(x): Convolution -> Convolution
        x = k.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            kernel_initializer=KERNEL_INITIALIZER,
            # padding="same",
        )(input_layer)
        x = k.layers.LeakyReLU(0.1)(x)

        x = k.layers.Conv2D(
            filters=128,
            kernel_size=4,
            strides=2,
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)
        x = k.layers.LeakyReLU(0.2)(x)
        x = k.layers.BatchNormalization()(x)

        x = k.layers.Conv2D(
            filters=512,
            kernel_size=4,
            strides=2,
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)
        x = k.layers.LeakyReLU(0.2)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Flatten()(x)

        # D(z): Dense
        z = k.layers.Dense(units=512, kernel_initializer=KERNEL_INITIALIZER)(
            input_encoded_layer
        )
        z = k.layers.LeakyReLU(0.2)(z)

        # Concatenate
        concat_input = k.layers.concatenate([x, z])

        # D(x, z): Dense -> Dense
        y = k.layers.Dense(units=1024, kernel_initializer=KERNEL_INITIALIZER)(
            concat_input
        )
        y = k.layers.LeakyReLU(0.2)(y)

        intermediate_layer = y

        output = k.layers.Dense(
            units=1, kernel_initializer=KERNEL_INITIALIZER, activation="sigmoid"
        )(y)

        self._discriminator = k.Model(
            inputs=[input_layer, input_encoded_layer],
            outputs=[output, intermediate_layer],
            name="bigan_discriminator",
        )

    def call(self, inputs, training=False):
        return self._discriminator(inputs, training=training)
