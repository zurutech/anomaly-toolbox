# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras as k

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


class Discriminator(k.Model):
    """
    Assemble the EGBAD BiGAN Discriminator as a :obj:`tf.keras.Model`.
    """

    def __init__(
        self,
        n_channels: int = 3,
        latent_space_dimension: int = 200,
        l2_penalty: float = 0.0,
    ):
        """
        Assemble the EGBAD BiGAN Discriminator as a :obj:`tf.keras.Model`.

        Args:
            n_channels: depth of the output image.
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
            l2_penalty: The penalty of the regularizer.

        Return:
            The assembled model.
        """

        super().__init__()

        input_dimension = (28, 28, n_channels)
        self._input_layer = k.layers.InputLayer(input_shape=input_dimension)
        self._input_encoded_layer = k.layers.InputLayer(
            input_shape=latent_space_dimension
        )
        self._concat = k.layers.Concatenate()

        # Backbone to elaborate the image shape data.
        self._backbone = k.Sequential(
            [
                self._input_layer,
                k.layers.Conv2D(
                    64,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    128,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    256,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    512,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    128,
                    (2, 2),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Flatten(),
            ]
        )

        # Part that elaborate the encoded data.
        self._encode_latent = k.Sequential(
            [
                k.layers.Dense(128, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
            ]
        )

        # Work on concatenated data.
        self._features = k.Sequential(
            [
                k.layers.Dense(1024, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.LeakyReLU(0.2),
                k.layers.Flatten(),
            ]
        )

        # Output layer.
        self._output = k.layers.Dense(
            1,
            kernel_initializer=KERNEL_INITIALIZER,
            use_bias=False,
        )

    def call(self, inputs, training=True):
        x, z = inputs

        d_x = self._backbone(x)
        d_z = self._encode_latent(z)

        concat_input = self._concat([d_x, d_z])

        features = self._features(concat_input)
        d_xz = self._output(features)

        return d_xz, features


class Decoder(k.Sequential):
    """
    Assemble the EGBAD BiGAN Decoder as a :obj:`tf.keras.Model`.
    """

    def __init__(
        self,
        n_channels: int = 3,
        latent_space_dimension: int = 200,
        l2_penalty: float = 0.0,
    ):
        """
        Assemble the EGBAD BiGAN Decoder as a :obj:`tf.keras.Model`.

        Note:
            This is designed to work with any image size in a fully-convolutional way.

        Args:
            n_channels: Depth of the output image.
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
            l2_penalty: The penalty for the kernel regularizer.
        """

        super().__init__(
            [
                k.layers.Input(shape=latent_space_dimension),
                k.layers.Dense(512, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.Dense(7 * 7 * 128, kernel_initializer=KERNEL_INITIALIZER),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Reshape(target_shape=(7, 7, 128)),
                k.layers.Conv2DTranspose(
                    256,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    128,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    n_channels,
                    (1, 1),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(1, 1),
                    padding="same",
                    use_bias=False,
                ),
                k.layers.Activation(k.activations.tanh),
            ]
        )


class Encoder(k.Sequential):
    """
    Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model`.
    """

    def __init__(
        self,
        n_channels: int = 3,
        latent_space_dimension: int = 200,
        l2_penalty: float = 0.0,
    ):
        """
        Assemble the EGBAD BiGAN Encoder as a :obj:`tf.keras.Model`.

        Note:
            This is designed to work with any image size in a fully-convolutional way.

        Args:
            n_channels: Depth of the input image.
            latent_space_dimension: Dimension (along the depth) of the resulting
                                    encoded representation.
            l2_penalty: The penalty for the kernel regularizer.

        """

        super().__init__(
            [
                k.layers.Input(shape=(28, 28, n_channels)),
                k.layers.Conv2D(
                    64,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    128,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    256,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    512,
                    (4, 4),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(
                    beta_initializer=ALMOST_ONE,
                    gamma_initializer=ALMOST_ONE,
                    momentum=0.1,
                    epsilon=1e-5,
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    latent_space_dimension,
                    (2, 2),
                    kernel_initializer=KERNEL_INITIALIZER,
                    strides=(1, 1),
                    padding="valid",
                    use_bias=False,
                ),  # 1
                k.layers.Flatten(),
            ]
        )
