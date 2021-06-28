"""Trainer for the GANomaly model."""

from typing import Dict, Optional, Set, Tuple
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses import ganomaly as losses
from anomaly_toolbox.models.ganomaly import (
    GANomalyAssembler,
    GANomalyDiscriminator,
    GANomalyGenerator,
)
from anomaly_toolbox.trainers.trainer import Trainer

__ALL__ = ["GANomaly"]


class GANomaly(Trainer):
    """GANomaly Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        input_dimension: Tuple[int, int, int],
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize GANomaly Networks."""
        print("Initializing GANomaly Trainer")
        super().__init__(
            dataset=dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        # Models
        # self.discriminator = GANomalyDiscriminator(input_dimension, filters)
        # self.generator = GANomalyGenerator(
        #     input_dimension, filters, self._hps["latent_vector_size"]
        # )

        filters = 64

        self.discriminator = GANomalyDiscriminator(
            input_dimension=input_dimension, filters=filters
        )
        self.generator = GANomalyGenerator(
            input_dimension=input_dimension,
            filters=filters,
            latent_space_dimension=self._hps["latent_vector_size"],
        )
        fake_batch_size = (1,) + input_dimension
        self.discriminator(tf.zeros(fake_batch_size))
        self.discriminator.summary()

        self.generator(tf.zeros(fake_batch_size))
        self.generator.summary()

        # Losses
        self._mse = tf.keras.losses.MeanSquaredError()
        self._mae = tf.keras.losses.MeanAbsoluteError()

        # Optimizers
        self.optimizer_ge = keras.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = keras.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # Training Metrics
        self.epoch_d_loss_avg = tf.keras.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_g_loss_avg = tf.keras.metrics.Mean(name="epoch_generator_loss")
        self.epoch_e_loss_avg = tf.keras.metrics.Mean(name="epoch_encoder_loss")
        self._training_keras_metrics = [
            self.epoch_d_loss_avg,
            self.epoch_g_loss_avg,
            self.epoch_e_loss_avg,
        ]
        # Test Metrics
        self.test_d_loss_avg = tf.keras.metrics.Mean(name="test_discriminator_loss")
        self.test_g_loss_avg = tf.keras.metrics.Mean(name="test_generator_loss")
        self.test_e_loss_avg = tf.keras.metrics.Mean(name="test_encoder_loss")
        self._test_keras_metrics = [
            self.test_d_loss_avg,
            self.test_g_loss_avg,
            self.test_e_loss_avg,
        ]

        self.keras_metrics = {
            metric.name: metric
            for metric in self._training_keras_metrics + self._test_keras_metrics
        }

    @staticmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        return {
            "learning_rate",
            "latent_vector_size",
            "adversarial_loss_weight",
            "contextual_loss_weight",
            "enc_loss_weight",
        }

    def train(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        step_log_frequency: int = 100,
        test_dataset: Optional[tf.data.Dataset] = None,
    ):
        for epoch in range(epoch):
            training_data, training_reconstructions = [], []
            batch_size = None
            for batch in dataset:
                if not batch_size:
                    batch_size = tf.shape(batch[0])[0]
                # Perform the train step
                x, x_hat, d_loss, g_loss, e_loss = self.step_fn(
                    batch,
                    adversarial_loss_weight,
                    contextual_loss_weight,
                    enc_loss_weight,
                    training=True,
                )

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                self.epoch_e_loss_avg.update_state(e_loss)
                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_ge.learning_rate.numpy()

                # Save the input images and their reconstructions for later use

                training_data.append(x)
                training_reconstructions.append(x_hat)

                if step % step_log_frequency == 0:
                    with self._summary_writer.as_default():
                        tf.summary.scalar("learning_rate", learning_rate, step=step)

                    tf.print(
                        "Step {:04d}: d_loss: {:.3f}, g_loss: {:.3f}, e_loss: {:.3f}, lr: {:.5f}".format(
                            step,
                            self.epoch_d_loss_avg.result(),
                            self.epoch_g_loss_avg.result(),
                            self.epoch_e_loss_avg.result(),
                            learning_rate,
                        )
                    )
            # |--------------------|
            # | Epoch-wise logging |
            # |--------------------|
            self.log(
                input_data=training_data[-1][:batch_size],
                reconstructions=training_reconstructions[-1][:batch_size],
                summary_writer=self._summary_writer,
                step=step,
                epoch=epoch,
                d_loss_metric=self.epoch_d_loss_avg,
                g_loss_metric=self.epoch_g_loss_avg,
                e_loss_metric=self.epoch_e_loss_avg,
                max_images_to_log=batch_size,
                training=True,
            )

            # |-----------------------|
            # | Perform the test step |
            # |-----------------------|
            if test_dataset:
                _, _, _ = self.test_phase(
                    test_dataset=test_dataset,
                    adversarial_loss_weight=adversarial_loss_weight,
                    contextual_loss_weight=contextual_loss_weight,
                    enc_loss_weight=enc_loss_weight,
                    epoch=epoch,
                    step=step,
                )
            # Reset metrics or the data will keep accruing becoming an average of ALL the epcohs
            self._reset_keras_metrics()

    def test_phase(
        self,
        test_dataset,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        step: int,
        epoch: int,
        log: bool = True,
    ) -> Tuple:
        """Perform the test pass on a given test_dataset."""
        test_iterator = iter(test_dataset)
        testing_data, testing_reconstructions = [], []
        batch_size = None
        for input_data in test_iterator:
            (test_x, test_x_hat, test_d_loss, test_g_loss, test_e_loss) = self.step_fn(
                input_data,
                adversarial_loss_weight,
                contextual_loss_weight,
                enc_loss_weight,
                training=False,
            )
            if not batch_size:
                batch_size = tf.shape(test_x)[0]
            testing_data.append(test_x)
            testing_reconstructions.append(test_x_hat)
            # Update the losses metrics
            self.test_d_loss_avg.update_state(test_d_loss)
            self.test_g_loss_avg.update_state(test_g_loss)
            self.test_e_loss_avg.update_state(test_e_loss)
        if log:
            self.log(
                input_data=testing_data[0][:batch_size],
                reconstructions=testing_reconstructions[0][:batch_size],
                summary_writer=self._summary_writer,
                step=step,
                epoch=epoch,
                d_loss_metric=self.test_d_loss_avg,
                g_loss_metric=self.test_g_loss_avg,
                e_loss_metric=self.test_e_loss_avg,
                max_images_to_log=batch_size,
                training=False,
            )
        return (
            self.test_d_loss_avg.result(),
            self.test_g_loss_avg.result(),
            self.test_e_loss_avg.result(),
        )

    @tf.function
    def step_fn(
        self,
        inputs,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        training: bool = True,
    ):
        """Single training step."""
        x, y = inputs
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction
            z, x_hat, z_hat = self.generator(x, training=training)

            # Discriminator on real data
            d_x, d_f_x = self.discriminator(x, training=training)

            # Discriminator on x_hat
            d_x_hat, d_f_x_hat = self.discriminator(x_hat, training=training)

            # g loss
            adversarial_loss = losses.adversarial_loss_fm(d_f_x, d_f_x_hat)
            e_loss = self._mse(z, z_hat)  # encoder loss
            l1_loss = self._mae(x, x_hat)  # contextual
            g_loss = (
                adversarial_loss_weight * adversarial_loss
                + contextual_loss_weight * l1_loss
                + enc_loss_weight * e_loss
            )

            # d loss
            d_loss = losses.discriminator_loss(d_x, d_x_hat)

        if training:
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)

            self.optimizer_ge.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
            self.optimizer_d.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )
        del tape
        return (
            x,
            x_hat,
            d_loss,
            g_loss,
            e_loss,
        )

    def log(
        self,
        input_data,
        reconstructions,
        summary_writer,
        step: int,
        epoch: int,
        d_loss_metric,
        g_loss_metric,
        e_loss_metric,
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
            e_loss_metric: Keras Metric
            max_images_to_log: Maximum amount of images that will be logged
            training: True for logging training, False for logging test epoch results

        """
        with summary_writer.as_default():
            hp.hparams(self._hps)
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
            tf.summary.scalar(
                "epoch_e_loss" if training else "test_epoch_e_loss",
                e_loss_metric.result(),
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
        print("--------------------------------")
        print(
            "{}: {:03d}: d_loss: {:.3f}, g_loss: {:.3f}, e_loss: {:.3f},".format(
                "EPOCH" if training else "TEST",
                epoch,
                d_loss_metric.result(),
                g_loss_metric.result(),
                e_loss_metric.result(),
            )
        )
        print("--------------------------------")

    # | ----------------- |
    # | Trainer functions |
    # | ----------------- |

    def train_mnist(
        self,
        epoch: int,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ) -> None:
        """
        Train GANomaly on MNIST dataset with one abnormal class.

        Args:
            epoch: Number of epochs.
            adversarial_loss_weight: weight for the adversarial loss.
            contextual_loss_weight: weight for the contextual loss (reconstruction loss).
            enc_loss_weight: weight for the encoder loss.
        """
        self.train(
            dataset=self._dataset.train_normal,
            epoch=epoch,
            adversarial_loss_weight=adversarial_loss_weight,
            contextual_loss_weight=contextual_loss_weight,
            enc_loss_weight=enc_loss_weight,
            test_dataset=self._dataset.test_normal,
        )
