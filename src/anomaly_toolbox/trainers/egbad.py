"""Trainer for the BiGAN model used in EGBAD."""

from typing import Dict, Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.losses import egbad as losses
from anomaly_toolbox.models import EGBADBiGANAssembler
from anomaly_toolbox.trainers.trainer import Trainer
from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset

__ALL__ = ["EGBAD"]


class EGBAD(Trainer):
    """EGBAD Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        input_dimension: Tuple[int, int, int],
        filters: int,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
    ):
        """Initialize EGBAD-BiGAN Networks."""
        print("Initializing EGBAD-BiGAN Trainer")
        super().__init__(dataset=dataset, hps=hps, summary_writer=summary_writer)

        # |--------|
        # | MODELS |
        # |--------|
        self.discriminator = EGBADBiGANAssembler.assemble_discriminator(
            input_dimension, filters, self._hps["latent_vector_size"]
        )
        self.generator = EGBADBiGANAssembler.assemble_decoder(
            input_dimension=self._hps["latent_vector_size"],
            output_dimension=input_dimension,
            filters=filters,
        )
        self.encoder = EGBADBiGANAssembler.assemble_encoder(
            input_dimension=input_dimension,
            filters=filters,
            latent_space_dimension=self._hps["latent_vector_size"],
        )

        fake_batch_size = (1,) + input_dimension
        fake_latent_vector = (1,) + (1, 1, self._hps["latent_vector_size"])
        self.generator(tf.zeros(fake_latent_vector))
        self.encoder(tf.zeros(fake_batch_size))
        self.discriminator([tf.zeros(fake_batch_size), tf.zeros(fake_latent_vector)])

        self.generator.summary()
        self.encoder.summary()
        self.discriminator.summary()

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
        self.optimizer_e = keras.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # |---------|
        # | Metrics |
        # |---------|
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
        self._keras_metrics = self._training_keras_metrics + self._test_keras_metrics

    def train(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
        step_log_frequency: int = 100,
        test_dataset: Optional[tf.data.Dataset] = None,
    ):
        for epoch in range(epoch):
            training_data, training_reconstructions, training_generated = [], [], []
            batch_size = None
            for batch in dataset:
                if not batch_size:
                    batch_size = tf.shape(batch[0])[0]
                # Perform the train step
                x, x_hat, xz_hat, d_loss, g_loss, e_loss = self.step_fn(
                    batch, training=True
                )

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                self.epoch_e_loss_avg.update_state(e_loss)
                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_g.learning_rate.numpy()

                # Save the input images and their reconstructions for later use
                training_data.append(x)
                training_generated.append(x_hat)
                training_reconstructions.append(xz_hat)

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
                generated=training_generated[-1][:batch_size],
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
                    epoch=epoch,
                    step=step,
                )
            # Reset metrics or the data will keep accruing becoming an average of ALL the epcohs
            self._reset_keras_metrics()

    def test_phase(
        self,
        test_dataset,
        step: int,
        epoch: int,
        log: bool = True,
    ) -> Tuple:
        """Perform the test pass on a given test_dataset."""
        test_iterator = iter(test_dataset)
        testing_data, testing_reconstructions, testing_generated = [], [], []
        batch_size = None
        for input_data in test_iterator:
            (
                test_x,
                test_x_hat,
                test_xz_hat,
                test_d_loss,
                test_g_loss,
                test_e_loss,
            ) = self.step_fn(input_data, training=False)
            if not batch_size:
                batch_size = tf.shape(test_x)[0]
            testing_data.append(test_x)
            testing_generated.append(test_x_hat)
            testing_reconstructions.append(test_xz_hat)
            # Update the losses metrics
            self.test_d_loss_avg.update_state(test_d_loss)
            self.test_g_loss_avg.update_state(test_g_loss)
            self.test_e_loss_avg.update_state(test_e_loss)
        if log:
            batch_size = batch_size.numpy()
            self.log(
                input_data=testing_data[0][:batch_size],
                reconstructions=testing_reconstructions[0][:batch_size],
                generated=testing_generated[0][:batch_size],
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
        training: bool = True,
    ):
        """Single training step."""
        x, y = inputs
        batch_size = tf.shape(x)[0]
        noise = tf.random.normal((batch_size, 1, 1, self._hps["latent_vector_size"]))
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction
            x_hat = self.generator(noise, training=training)

            z = self.encoder(x, training=training)
            xz_hat = self.generator(z, training=training)

            d_x, x_features = self.discriminator([x, z], training=training)
            d_x_hat, x_hat_features = self.discriminator(
                [x_hat, noise], training=training
            )

            # Losses
            d_loss = losses.discriminator_loss(d_x, d_x_hat)
            g_loss = losses.adversarial_loss_fm(x_hat_features, x_features)
            e_loss = losses.encoder_loss(d_x)

        if training:
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            e_grads = tape.gradient(e_loss, self.encoder.trainable_variables)

            self.optimizer_d.apply_gradients(
                zip(d_grads, self.discriminator.trainable_variables)
            )
            self.optimizer_g.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
            self.optimizer_e.apply_gradients(
                zip(e_grads, self.encoder.trainable_variables)
            )
        del tape

        return (
            x,
            x_hat,
            xz_hat,
            d_loss,
            g_loss,
            e_loss,
        )

    def log(
        self,
        input_data,
        reconstructions,
        generated,
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
            generated: Generated images from noise
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
                "training_generations" if training else "test_generations",
                generated,
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

    # | ----------------- |
    # | Trainer functions |
    # | ----------------- |

    def train_mnist(
        self,
        epoch: int,
    ) -> None:
        """
        Train EGBAD on MNIST dataset with one abnormal class.

        Args:
            epoch: Number of epochs.
        """

        self.train(
            dataset=self._dataset.train_normal,
            epoch=epoch,
            test_dataset=self._dataset.test_normal,
        )
