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

import tensorflow.keras as k

KERNEL_INITIALIZER = k.initializers.RandomNormal(mean=0.0, stddev=0.02)
ALMOST_ONE = k.initializers.RandomNormal(mean=1.0, stddev=0.02)


class Decoder(k.Sequential):
    def __init__(
        self,
        n_channels: int = 3,
        latent_space_dimension: int = 100,
        l2_penalty: float = 0.0,
    ):
        super().__init__(
            [
                k.layers.InputLayer(input_shape=(latent_space_dimension,)),
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
                    filters=256,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2DTranspose(
                    filters=n_channels,
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.Activation(k.activations.tanh),
            ]
        )


class Encoder(k.Sequential):
    def __init__(
        self,
        n_channels: int = 3,
        latent_space_dimension: int = 100,
        l2_penalty: float = 0.0,
    ):
        super().__init__(
            [
                k.layers.InputLayer(input_shape=(32, 32, n_channels)),
                k.layers.Conv2D(
                    filters=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=256,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.BatchNormalization(),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=latent_space_dimension,
                    kernel_size=(4, 4),
                    strides=(1, 1),
                    padding="valid",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.Flatten(),
            ]
        )


class Discriminator(k.Model):
    def __init__(
        self,
        n_channels: int = 3,
        l2_penalty: float = 0.0,
    ):
        super().__init__()

        # Input
        input_dimension = (32, 32, n_channels)
        self._input_layer = k.layers.InputLayer(input_shape=input_dimension)

        self._backbone = k.Sequential(
            [
                self._input_layer,
                k.layers.Conv2D(
                    filters=32,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=64,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=128,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Conv2D(
                    filters=256,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    padding="same",
                    kernel_initializer=KERNEL_INITIALIZER,
                    use_bias=False,
                    kernel_regularizer=k.regularizers.l2(l2_penalty),
                ),
                k.layers.LeakyReLU(0.2),
                k.layers.Flatten(),
            ]
        )

        self._output = k.layers.Dense(
            1,
            kernel_initializer=KERNEL_INITIALIZER,
            use_bias=False,
            # kernel_regularizer=k.regularizers.l2(l2_penalty),
        )

    def call(self, inputs, training=True, mask=None):

        features = self._backbone(inputs)
        output = self._output(features)

        return output, features
