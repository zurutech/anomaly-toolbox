"""BiGAN Architecture Implementation as used in EGBAD."""

import tensorflow.keras as k
import tensorflow as tf

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


class Encoder(k.Model):
    """
    Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 128):
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
            filters,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
        )(input_layer)
        x = k.layers.LeakyReLU(alpha=0.2)(x)

        side = 16
        while side > 4:
            filters *= 2
            x = k.layers.Conv2D(
                filters,
                kernel_size=4,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_regularizer=k.regularizers.l2(l2_penalty),
                kernel_initializer=KERNEL_INITIALIZER,
            )(x)
            x = k.layers.BatchNormalization(
                beta_initializer=ALMOST_ONE,
                gamma_initializer=ALMOST_ONE,
                momentum=0.1,
                epsilon=1e-5,
            )(x)
            x = k.layers.LeakyReLU(alpha=0.2)(x)
            side //= 2

        x = k.layers.Conv2D(
            latent_space_dimension,
            kernel_size=4,
            strides=1,
            padding="valid",
            use_bias=False,
            kernel_initializer=KERNEL_INITIALIZER,
        )(x)

        self._encoder = k.Model(input_layer, x, name="bigan_encoder")

    def call(self, inputs, training=False):
        return self._encoder(inputs, training=training)


class Decoder(k.Model):
    """
    Assemble the EGBAD BiGAN Decoder as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 128):
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

        input_layer = k.layers.Input(shape=(1, 1, latent_space_dimension))

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
            1,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=k.regularizers.l2(l2_penalty),
            kernel_initializer=KERNEL_INITIALIZER,
            activation="tanh",
        )(x)

        self._decoder = k.Model(input_layer, x, name="bigan_decoder")

    def call(self, inputs, training=False):
        return self._decoder(inputs, training=training)


class Discriminator(k.Model):
    """
    Assemble the EGBAD BiGAN Discriminator as a :obj:`tf.keras.Model`.
    """

    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 128):
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
        input_layer = k.layers.Input(shape=input_dimension)
        input_encoding = k.layers.Input(shape=(1, 1, latent_space_dimension))
        d_z = k.layers.Conv2D(
            128,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(input_encoding)
        d_z = k.layers.LeakyReLU(0.2)(d_z)  # D(z) <-> 1x1x128

        concat_input = k.layers.concatenate(
            [encoder(input_layer), d_z]
        )  # D(x|z) <-> 1x1x256

        fc = k.layers.Conv2D(
            1024,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(concat_input)
        feature = k.layers.LeakyReLU(0.2, name="feature")(fc)  # 1x1x1024

        out = k.layers.Conv2D(
            1,
            (1, 1),
            kernel_initializer=KERNEL_INITIALIZER,
            strides=(1, 1),
            padding="valid",
            use_bias=False,
        )(
            feature
        )  # 1x1x1

        self._discriminator = k.Model(
            inputs=[input_layer, input_encoding],
            outputs=[out, feature],
            name="bigan_discriminator",
        )

    def call(self, inputs, training=False):
        return self._discriminator(inputs, training=training)
