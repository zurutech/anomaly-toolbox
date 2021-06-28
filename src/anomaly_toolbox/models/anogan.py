"""Implementation of the 28x28 input resolution models of AnoGAN."""

import tensorflow as tf
import tensorflow.keras as k


class Generator(k.Model):
    """Generator in fully convolutional fashion.
    Input: 1x1x input_dimension.
    """

    def __init__(self, input_dimension: int):
        """Generator model.
        Args:
            input_dimension: the dimension of the latent vector.
        """

        super().__init__()
        input_layer = k.layers.Input(shape=(1, 1, input_dimension))

        x = k.layers.Dense(7 * 7 * 128)(input_layer)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.LeakyReLU(0.2)(x)
        x = k.layers.Reshape((7, 7, 128))(x)

        x = k.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)
        x = k.layers.Conv2D(64, (2, 2), padding="same")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)

        x = k.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)
        x = k.layers.Conv2D(1, (5, 5), padding="same")(x)
        x = k.layers.Activation("tanh")(x)
        self._model = k.Model(input_layer, x, name="anogan_mnist_generator")

    def call(self, inputs, training=False) -> tf.Tensor:
        """Forward pass.
        Args:
            inputs: input batch
            training: toggle the model status from training to inference.
        Returns:
            The generated output in [-1,1].
        """
        return self._model(inputs, training)


class Discriminator(k.Model):
    """Discriminator model. Expects a batch of 28x28x1 input images."""

    def __init__(self):
        super().__init__()

        input_layer = k.layers.Input(shape=(28, 28, 1))

        x = k.layers.Conv2D(64, (5, 5), padding="same")(input_layer)
        x = k.layers.LeakyReLU(0.2)(x)
        features = k.layers.MaxPool2D(pool_size=(2, 2))(x)
        # NOTE: https://github.com/tkwoo/anogan-keras/blob/master/anogan.py

        x = k.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same")(features)
        x = k.layers.LeakyReLU(0.2)(x)
        x = k.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = k.layers.Flatten()(x)
        x = k.layers.Dense(1)(x)

        self._model = k.Model(
            input_layer, outputs=[x, features], name="anogan_mninst_discriminator"
        )

    def call(self, inputs, training=False) -> tf.Tensor:
        """Forward pass.
        Args:
            inputs: input batch
            training: toogle the model status from training to inference.
        Returns:
            The discriminator decision (single neuron, linear activation).
        """

        return self._model(inputs, training)
