"""BiGAN Architecture Implementation as used in EGBAD."""
import tensorflow.keras as k

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


class Encoder(k.Sequential):
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
        super().__init__(
            [
                k.layers.Input(shape=(28, 28, n_channels)),
                k.layers.Conv2D(
                    32,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    kernel_regularizer=k.regularizers.l2(l2=0.0),
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(alpha=0.2),
                k.layers.Conv2D(
                    64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_regularizer=k.regularizers.l2(l2=0.0),
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(alpha=0.2),
                k.layers.BatchNormalization(),
                k.layers.Conv2D(
                    128,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_regularizer=k.regularizers.l2(l2=0.0),
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(alpha=0.2),
                k.layers.BatchNormalization(),
                k.layers.Flatten(),
                k.layers.Dense(
                    units=latent_space_dimension, kernel_initializer=KERNEL_INITIALIZER
                ),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training=training, mask=mask)


class Decoder(k.Sequential):
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
        # output dimension is 28, 28
        l2_penalty = 0.0
        super().__init__(
            [
                k.layers.Input(shape=(latent_space_dimension,)),
                k.layers.Dense(1024, activation="relu"),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.Dense(7 * 7 * 128, activation="relu"),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.Reshape(target_shape=(7, 7, 128)),
                k.layers.Conv2DTranspose(
                    64,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                    kernel_initializer=KERNEL_INITIALIZER,
                    activation="relu",
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.Conv2DTranspose(
                    n_channels,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                    kernel_initializer=KERNEL_INITIALIZER,
                    activation="tanh",
                ),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training=training, mask=mask)


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

        # Input layers
        self._input_layer = k.layers.Input(shape=input_dimension)
        self._input_encoded_layer = k.layers.Input(shape=(latent_space_dimension,))

        # D(x): Convolution -> Convolution
        self._backbone = k.Sequential(
            [
                k.layers.Input(shape=input_dimension),
                k.layers.Conv2D(
                    filters=64,
                    kernel_size=4,
                    strides=2,
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(0.1),
                k.layers.Conv2D(
                    filters=128,
                    kernel_size=4,
                    strides=2,
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.BatchNormalization(),
                k.layers.Conv2D(
                    filters=512,
                    kernel_size=4,
                    strides=2,
                    kernel_initializer=KERNEL_INITIALIZER,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.BatchNormalization(),
                k.layers.Flatten(),
            ]
        )

        # D(z): Dense
        self._encode_latent = k.Sequential(
            [
                k.layers.Dense(units=512, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
            ]
        )

        # D(x, z): Dense -> Dense
        self._features = k.Sequential(
            [
                k.layers.Dense(units=1024, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
            ]
        )

        self._output = k.layers.Dense(
            units=1, kernel_initializer=KERNEL_INITIALIZER, activation="sigmoid"
        )

    def call(self, inputs, training=None, mask=None):
        x, z = inputs

        d_x = self._backbone(x)
        d_z = self._encode_latent(z)

        concat_input = k.layers.concatenate([d_x, d_z])

        features = self._features(concat_input)
        d_xz = self._output(features)

        return d_xz, features
