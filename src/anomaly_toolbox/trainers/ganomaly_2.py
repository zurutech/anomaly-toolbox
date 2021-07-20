"""Trainer for the GANomaly model."""

from typing import Dict, Optional, Set, Tuple
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses.ganomaly import AdversarialLoss, generator_bce
from anomaly_toolbox.models.ganomaly_2 import Discriminator, Encoder, Decoder
from anomaly_toolbox.trainers.trainer import Trainer


class GANomaly(Trainer):
    """GANomaly Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        # input_dimension: Tuple[int, int, int],
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize GANomaly Networks."""
        print("Initializing GANomaly Trainer")
        super().__init__(
            dataset=dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        n_channels = dataset.channels

        # Discriminator
        self.discriminator = Discriminator(n_channels=n_channels, l2_penalty=0.2)

        # Generator (aka Decoder)
        self.generator = Decoder(
            n_channels=n_channels,
            latent_space_dimension=self._hps["latent_vector_size"],
            l2_penalty=0.2,
        )

        # Encoder
        self.encoder = Encoder(
            n_channels=n_channels,
            latent_space_dimension=self._hps["latent_vector_size"],
            l2_penalty=0.2,
        )

        fake_batch_size = (1, 32, 32, 1)
        self.discriminator(tf.zeros(fake_batch_size))
        self.discriminator.summary()

        self.encoder(tf.zeros(fake_batch_size))
        self.encoder.summary()

        fake_latent_vector = (1, self._hps["latent_vector_size"])
        self.generator(tf.zeros(fake_latent_vector))
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
        # self.epoch_e_loss_avg = tf.keras.metrics.Mean(name="epoch_encoder_loss")
        self._training_keras_metrics = [
            self.epoch_d_loss_avg,
            self.epoch_g_loss_avg,
            # self.epoch_e_loss_avg,
        ]
        # Test Metrics
        self.test_d_loss_avg = tf.keras.metrics.Mean(name="test_discriminator_loss")
        self.test_g_loss_avg = tf.keras.metrics.Mean(name="test_generator_loss")
        # self.test_e_loss_avg = tf.keras.metrics.Mean(name="test_encoder_loss")
        self._test_keras_metrics = [
            self.test_d_loss_avg,
            self.test_g_loss_avg,
            # self.test_e_loss_avg,
        ]

        self.keras_metrics = {
            metric.name: metric
            for metric in self._training_keras_metrics + self._test_keras_metrics
        }

        self._minmax = AdversarialLoss(from_logits=True)

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
        epochs: int,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        step_log_frequency: int = 100,
        test_dataset: Optional[tf.data.Dataset] = None,
    ):
        for epoch in range(epochs):

            for batch in self._dataset.train_normal:
                x, _ = batch

                # Perform the train step
                g_z, g_ex, d_loss, g_loss, e_loss = self.train_step(
                    x,
                    adversarial_loss_weight,
                    contextual_loss_weight,
                    enc_loss_weight,
                )

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)

                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_ge.learning_rate.numpy()

                if tf.equal(tf.math.mod(step, step_log_frequency), 0):
                    with self._summary_writer.as_default():
                        tf.summary.scalar("learning_rate", learning_rate, step=step)
                        tf.summary.image(
                            "x/g_z/g_ex",
                            tf.concat([x, g_z, g_ex], axis=2),
                            step=step,
                        )
                        tf.summary.scalar(
                            "d_loss",
                            self.epoch_d_loss_avg.result(),
                            step=step,
                        )
                        tf.summary.scalar(
                            "g_loss",
                            self.epoch_g_loss_avg.result(),
                            step=step,
                        )

                    tf.print(
                        "Step {:04d}: d_loss: {:.3f}, ge_loss: {:.3f},"
                        "lr: {:.5f}".format(
                            step,
                            self.epoch_d_loss_avg.result(),
                            self.epoch_g_loss_avg.result(),
                            # self.epoch_e_loss_avg.result(),
                            learning_rate,
                        )
                    )

            # Epoch end
            tf.print(epoch, "Epoch completed")

            # |--------------------|
            # | Epoch-wise logging |
            # |--------------------|
            # self.log(
            #     input_data=training_data[-1][:batch_size],
            #     reconstructions=training_reconstructions[-1][:batch_size],
            #     summary_writer=self._summary_writer,
            #     step=step,
            #     epoch=epoch,
            #     d_loss_metric=self.epoch_d_loss_avg,
            #     g_loss_metric=self.epoch_g_loss_avg,
            #     e_loss_metric=self.epoch_e_loss_avg,
            #     max_images_to_log=batch_size,
            #     training=True,
            # )

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
            # Reset metrics or the data will keep accruing becoming an average of ALL the epochs
            self._reset_keras_metrics()

    # def test_phase(
    #     self,
    #     test_dataset,
    #     adversarial_loss_weight: float,
    #     contextual_loss_weight: float,
    #     enc_loss_weight: float,
    #     step: int,
    #     epoch: int,
    #     log: bool = True,
    # ) -> Tuple:
    #     """Perform the test pass on a given test_dataset."""
    #     test_iterator = iter(test_dataset)
    #     testing_data, testing_reconstructions = [], []
    #     batch_size = None
    #     for input_data in test_iterator:
    #         (
    #             test_x,
    #             test_x_hat,
    #             test_d_loss,
    #             test_g_loss,
    #             test_e_loss,
    #         ) = self.train_step(
    #             input_data,
    #             adversarial_loss_weight,
    #             contextual_loss_weight,
    #             enc_loss_weight,
    #             training=False,
    #         )
    #         if not batch_size:
    #             batch_size = tf.shape(test_x)[0]
    #         testing_data.append(test_x)
    #         testing_reconstructions.append(test_x_hat)
    #         # Update the losses metrics
    #         self.test_d_loss_avg.update_state(test_d_loss)
    #         self.test_g_loss_avg.update_state(test_g_loss)
    #         self.test_e_loss_avg.update_state(test_e_loss)
    #     if log:
    #         self.log(
    #             input_data=testing_data[0][:batch_size],
    #             reconstructions=testing_reconstructions[0][:batch_size],
    #             summary_writer=self._summary_writer,
    #             step=step,
    #             epoch=epoch,
    #             d_loss_metric=self.test_d_loss_avg,
    #             g_loss_metric=self.test_g_loss_avg,
    #             e_loss_metric=self.test_e_loss_avg,
    #             max_images_to_log=batch_size,
    #             training=False,
    #         )
    #     return (
    #         self.test_d_loss_avg.result(),
    #         self.test_g_loss_avg.result(),
    #         self.test_e_loss_avg.result(),
    #     )

    # @tf.function
    def train_step(
        self,
        x,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):

        # Random noise
        z = tf.random.normal((tf.shape(x)[0], self._hps["latent_vector_size"]))

        """Single training step."""
        with tf.GradientTape(persistent=True) as tape:
            # Generator reconstruction from random noise
            g_z = self.generator(z, training=True)  # TODO or False?

            # Discriminator on real data
            d_x, d_x_features = self.discriminator(x, training=True)

            # Reconstruct real data after encoding
            e_x = self.encoder(x, training=True)
            g_ex = self.generator(e_x, training=True)

            # Discriminator on the reconstructed real data g_ex
            d_gex, d_gex_features = self.discriminator(inputs=g_ex, training=True)

            # Encode the reconstructed real data g_ex
            e_gex = self.encoder(g_ex, training=True)

            # Discriminator Loss
            # d_loss = self._minmax(d_x_features, d_gex_features)
            d_loss = self._minmax(d_x, d_gex)

            # Generator Loss
            # adversarial_loss = losses.adversarial_loss_fm(d_f_x, d_f_x_hat)
            bce_g_loss = generator_bce(g_ex, from_logits=True)

            l1_loss = self._mae(x, g_ex)  # contextual
            e_loss = self._mse(e_x, e_gex)  # encoder loss

            g_loss = (
                adversarial_loss_weight * bce_g_loss
                + contextual_loss_weight * l1_loss
                + enc_loss_weight * e_loss
            )

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        del tape

        self.optimizer_ge.apply_gradients(
            zip(
                g_grads,
                self.generator.trainable_variables + self.encoder.trainable_variables,
            )
        )
        self.optimizer_d.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        # TODO: If d_loss = self._minmax(d_x_features, d_gex_features), dense layer would return
        # a warning. To suppress the warining "Gradients does not exist for variables" try using
        # the following lines.
        # self.optimizer_d.apply_gradients(
        #     (grad, var)
        #     for (grad, var) in zip(d_grads, self.discriminator.trainable_variables)
        #     if grad is not None
        # )

        return (
            g_z,
            g_ex,
            d_loss,
            g_loss,
            e_loss,
        )
