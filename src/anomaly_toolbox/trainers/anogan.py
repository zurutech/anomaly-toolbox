"""Trainer for the AnoGAN model."""

from typing import Dict, Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets import MNISTDataset
from anomaly_toolbox.losses import adversarial_loss, feature_matching_loss
from anomaly_toolbox.models import AnoGANAssembler, AnoGANMNISTAssembler

from .interface import Trainer

__ALL__ = ["AnoGAN", "AnoGANMNIST"]


class AnoGAN(Trainer):
    """GANomaly Trainer."""

    def __init__(
        self,
        input_dimension: Tuple[int, int, int],
        filters: int,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        use_generic_architecture: bool = True,
    ):
        """Initialize AnoGAN Trainer."""
        super().__init__(hps=hps, summary_writer=summary_writer)
        # |--------|
        # | MODELS |
        # |--------|
        if use_generic_architecture:
            print("Initializing AnoGAN Trainer")
            self.discriminator = AnoGANAssembler.assemble_discriminator(
                input_dimension=input_dimension, filters=filters
            )
            self.generator = AnoGANAssembler.assemble_generator(
                input_dimension=hps["latent_vector_size"],
                output_dimension=input_dimension,
                filters=filters,
            )
            self._validate_models(input_dimension, hps["latent_vector_size"])

        # |------------|
        # | OPTIMIZERS |
        # |------------|
        # TODO: These should be constructed from passed HPS
        self.optimizer_g = keras.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = keras.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # |---------|
        # | Metrics |
        # |---------|

        # Training Metrics
        self.epoch_d_loss_avg = tf.keras.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_g_loss_avg = tf.keras.metrics.Mean(name="epoch_generator_loss")
        self._training_keras_metrics = [self.epoch_d_loss_avg, self.epoch_g_loss_avg]

        # Test Metrics
        self.test_d_loss_avg = tf.keras.metrics.Mean(name="test_discriminator_loss")
        self.test_g_loss_avg = tf.keras.metrics.Mean(name="test_generator_loss")

        self._test_keras_metrics = [self.test_d_loss_avg, self.test_g_loss_avg]
        self._keras_metrics = self._training_keras_metrics + self._test_keras_metrics

    def _validate_models(
        self, input_dimension: Tuple[int, int, int], latent_vector_size: int
    ):
        fake_latent_vector = (1,) + (1, 1, latent_vector_size)
        self.generator(tf.zeros(fake_latent_vector))
        self.generator.summary()

        fake_batch_size = (1,) + input_dimension
        self.discriminator(tf.zeros(fake_batch_size))
        self.discriminator.summary()

    def train(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        epochs: int,
        step_log_frequency: int = 100,
        test_dataset: Optional[tf.data.Dataset] = None,
    ):
        for epoch in range(epochs):
            training_data, training_reconstructions = [], []
            for batch in dataset:
                # Perform the train step
                x, x_hat, d_loss, g_loss = self.step_fn(
                    batch, batch_size=batch_size, training=True
                )

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_g.learning_rate.numpy()

                # Save the input images and their reconstructions for later use
                training_data.append(x)
                training_reconstructions.append(x_hat)

                if step % step_log_frequency == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar("learning_rate", learning_rate, step=step)

                    tf.print(
                        "Step {:04d}: d_loss: {:.3f}, g_loss: {:.3f}, lr: {:.5f}".format(
                            step,
                            self.epoch_d_loss_avg.result(),
                            self.epoch_g_loss_avg.result(),
                            learning_rate,
                        )
                    )

            # |--------------------|
            # | Epoch-wise logging |
            # |--------------------|
            self.log(
                input_data=training_data[-1][:batch_size],
                reconstructions=training_reconstructions[-1][:batch_size],
                summary_writer=self.summary_writer,
                step=step,
                epoch=epoch,
                d_loss_metric=self.epoch_d_loss_avg,
                g_loss_metric=self.epoch_g_loss_avg,
                max_images_to_log=batch_size,
                training=True,
            )

            # # |-----------------------|
            # # | Perform the test step |
            # # |-----------------------|
            if test_dataset:
                _, _ = self.test_phase(
                    test_dataset=test_dataset,
                    batch_size=batch_size,
                    epoch=epoch,
                    step=step,
                )
            # Reset metrics or the data will keep accruing becoming an average of ALL the epcohs
            self._reset_keras_metrics()

    def test_phase(
        self,
        test_dataset,
        batch_size: int,
        step: int,
        epoch: int,
        log: bool = True,
    ) -> Tuple:
        """Perform the test pass on a given test_dataset."""
        test_iterator = iter(test_dataset)
        testing_data, testing_reconstructions = [], []
        for input_data in test_iterator:
            (test_x, test_x_hat, test_d_loss, test_g_loss) = self.step_fn(
                input_data, batch_size=batch_size, training=False
            )
            testing_data.append(test_x)
            testing_reconstructions.append(test_x_hat)
            # Update the losses metrics
            self.test_d_loss_avg.update_state(test_d_loss)
            self.test_g_loss_avg.update_state(test_g_loss)
        if log:
            self.log(
                input_data=testing_data[0][:batch_size],
                reconstructions=testing_reconstructions[0][:batch_size],
                summary_writer=self.summary_writer,
                step=step,
                epoch=epoch,
                d_loss_metric=self.test_d_loss_avg,
                g_loss_metric=self.test_g_loss_avg,
                max_images_to_log=batch_size,
                training=False,
            )
        return self.test_d_loss_avg.result(), self.test_g_loss_avg.result()

    @tf.function
    def step_fn(
        self,
        inputs,
        batch_size,
        training: bool = True,
    ):
        """Single training step."""
        noise = tf.random.normal((batch_size, 1, 1, self.hps["latent_vector_size"]))
        x, y = inputs
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction
            x_hat = self.generator(noise, training=training)

            d_x, x_features = self.discriminator(x, training=training)
            d_x_hat, x_hat_features = self.discriminator(x_hat, training=training)

            # Losses
            d_loss = adversarial_loss(d_x, d_x_hat)
            g_loss = feature_matching_loss(x_hat_features, x_features)

        if training:
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)

            self.optimizer_d.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )
            self.optimizer_g.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
        del tape

        return x, x_hat, d_loss, g_loss

    def log(
        self,
        input_data,
        reconstructions,
        summary_writer,
        step: int,
        epoch: int,
        d_loss_metric,
        g_loss_metric,
        max_images_to_log: int,
        training: bool = True,
    ) -> None:
        """
        Log data (images, losses, learning rate) to TensorBoard.

        Args:
            input_data: Input images
            reconstructions: Reconstructions
            summary_writer: TensorFlow SummaryWriter to use for logging
            step: Current step
            epoch: Current epoch
            d_loss_metric: Keras Metric
            g_loss_metric: Keras Metric
            max_images_to_log: Maximum amount of images that will be logged
            training: True for logging training, False for logging test epoch results

        """
        with summary_writer.as_default():
            hp.hparams(self.hps)
            # |-----------------|
            # | Logging scalars |
            # |-----------------|
            tf.summary.scalar(
                "epoch_d_loss" if training else "test_epoch_d_loss",
                d_loss_metric.result(),
                step=step,
            )
            tf.summary.scalar(
                "epoch_g_loss" if training else "test_epoch_g_loss",
                g_loss_metric.result(),
                step=step,
            )
            # |----------------|
            # | Logging images |
            # |----------------|
            tf.summary.image(
                "training_data" if training else "test_data",
                input_data,
                max_outputs=max_images_to_log,
                step=step,
            )
            tf.summary.image(
                "training_reconstructions" if training else "test_reconstructions",
                reconstructions,
                max_outputs=max_images_to_log,
                step=step,
            )
        # -----
        tf.print("--------------------------------")
        tf.print(
            "{}: {}: d_loss: {:.3f}, g_loss: {:.3f}".format(
                "EPOCH" if training else "TEST",
                epoch,
                d_loss_metric.result(),
                g_loss_metric.result(),
            )
        )
        tf.print("--------------------------------")

    # | ----------------- |
    # | Trainer functions |
    # | ----------------- |

    # def train_mnist(
    #     self,
    #     batch_size: int,
    #     epoch: int,
    #     anomalous_label: int,
    # ) -> None:
    #     """
    #     Train AnoGAN on MNIST dataset with one abnormal class.

    #     Args:
    #         batch_size:
    #         epoch:
    #         anomalous_label:
    #         adversarial_loss_weight: weight for the adversarial loss
    #         contextual_loss_weight: weight for the contextual loss (reconstruction loss)
    #         enc_loss_weight: weight for the encoder loss
    #     """
    #     ds_builder = MNISTDataset()
    #     (
    #         self.ds_train,
    #         self.ds_train_anomalous,
    #         self.ds_test,
    #         self.ds_test_anomalous,
    #     ) = ds_builder.assemble_datasets(
    #         anomalous_label=anomalous_label, batch_size=batch_size, new_size=(32, 32)
    #     )
    #     self.train(
    #         dataset=self.ds_train,
    #         batch_size=batch_size,
    #         epoch=epoch,
    #         test_dataset=self.ds_test,
    #     )


class AnoGANMNIST(AnoGAN):
    input_dimension: Tuple[int, int, int] = (28, 28, 1)

    def __init__(self, hps: Dict, summary_writer: tf.summary.SummaryWriter):
        super().__init__(
            use_generic_architecture=False,  # Model Architectures will be manually initialized
            input_dimension=self.input_dimension,  # Unused
            filters=0,  # Unused
            hps=hps,
            summary_writer=summary_writer,
        )
        print("Initializing AnoGANMNIST (hardcoded architecture) Trainer")
        self.discriminator = AnoGANMNISTAssembler.assemble_discriminator()
        self.generator = AnoGANMNISTAssembler.assemble_generator(
            input_dimension=hps["latent_vector_size"]
        )
        self._validate_models(self.input_dimension, hps["latent_vector_size"])

    def train_mnist(
        self,
        batch_size: int,
        epochs: int,
        anomalous_label: int,
    ) -> None:
        """
        Train AnoGAN on MNIST dataset with one abnormal class.

        Args:
            batch_size:
            epochs:
            anomalous_label:
            adversarial_loss_weight: weight for the adversarial loss
            contextual_loss_weight: weight for the contextual loss (reconstruction loss)
            enc_loss_weight: weight for the encoder loss
        """
        ds_builder = MNISTDataset()
        (
            self.ds_train,
            self.ds_train_anomalous,
            self.ds_test,
            self.ds_test_anomalous,
        ) = ds_builder.assemble_datasets(
            anomalous_label=anomalous_label, batch_size=batch_size, drop_remainder=True
        )
        self.train(
            dataset=self.ds_train,
            batch_size=batch_size,
            epochs=epochs,
            test_dataset=self.ds_test,
        )
