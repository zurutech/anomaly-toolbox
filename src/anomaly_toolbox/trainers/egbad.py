"""Trainer for the BiGAN model used in EGBAD."""

from pathlib import Path
from typing import Dict, Set, Tuple
import json

import tensorflow as tf
import tensorflow.keras as k

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses.egbad import (
    adversarial_loss_fm,
    discriminator_loss,
    encoder_loss,
    residual_loss,
)
from anomaly_toolbox.models.egbad import Decoder, Discriminator, Encoder
from anomaly_toolbox.trainers.trainer import Trainer


class EGBAD(Trainer):
    """EGBAD Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        input_dimension: Tuple[int, int, int],
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize EGBAD-BiGAN Networks."""
        super().__init__(
            dataset=dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        # Models
        n_channels = tf.shape(next(iter(dataset.train.take(1)))[0])[-1]
        self.discriminator = Discriminator(n_channels, self._hps["latent_vector_size"])
        self.encoder = Encoder(n_channels, self._hps["latent_vector_size"])
        self.generator = Decoder(n_channels, self._hps["latent_vector_size"])

        # Instantiate and define with correct input shape
        fake_batch_size = (1,) + input_dimension
        fake_latent_vector = (1,) + (1, 1, self._hps["latent_vector_size"])
        self.generator(tf.zeros(fake_latent_vector))
        self.encoder(tf.zeros(fake_batch_size))
        self.discriminator([tf.zeros(fake_batch_size), tf.zeros(fake_latent_vector)])

        self.generator.summary()
        self.encoder.summary()
        self.discriminator.summary()

        # Optimizers
        self.optimizer_g = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.optimizer_e = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # Training Metrics
        self.epoch_d_loss_avg = k.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_g_loss_avg = k.metrics.Mean(name="epoch_generator_loss")
        self.epoch_e_loss_avg = k.metrics.Mean(name="epoch_encoder_loss")
        self._auprc = k.metrics.AUC(name="auprc", curve="PR", num_thresholds=500)
        self.keras_metrics = {
            metric.name: metric
            for metric in [
                self.epoch_d_loss_avg,
                self.epoch_g_loss_avg,
                self.epoch_e_loss_avg,
                self._auprc,
            ]
        }

        self._flatten = k.layers.Flatten()

        self._alpha = tf.constant(0.9)

    @staticmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        return {"learning_rate", "latent_vector_size"}

    def train(
        self,
        epochs: int,
        step_log_frequency: int = 100,
    ):
        for epoch in tf.range(epochs):
            for batch in self._dataset.train:
                x, _ = batch

                # Perform the train step
                g_z, xz_hat, d_loss, g_loss, e_loss = self.train_step(batch)

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                self.epoch_e_loss_avg.update_state(e_loss)
                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_g.learning_rate.numpy()

                if tf.equal(tf.math.mod(step, step_log_frequency), 0):
                    with self._summary_writer.as_default():
                        tf.summary.scalar("learning_rate", learning_rate, step=step)
                        tf.summary.image(
                            "x_xhat_xz_hat",
                            tf.concat([x, g_z, xz_hat], axis=2),
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
                        tf.summary.scalar(
                            "e_loss",
                            self.epoch_e_loss_avg.result(),
                            step=step,
                        )

                    tf.print(
                        "Step {:04d}: d_loss: {:.3f}, g_loss: {:.3f}, e_loss: {:.3f}, lr: {:.5f}".format(
                            step,
                            self.epoch_d_loss_avg.result(),
                            self.epoch_g_loss_avg.result(),
                            self.epoch_e_loss_avg.result(),
                            learning_rate,
                        )
                    )
            # Epoch end
            tf.print(epoch, " completed")

            # Model selection at the end of every epoch

            # Calculate AUPRC - Area Under Precision Recall Curve
            # 1. Compute the the anomaly score
            # 2. Use the AUC object to compute the AUCROC with different
            # thresholds values on the anomaly score.
            self._auprc.reset_state()
            best_auprc = -1
            for batch in self._dataset.test:
                x, labels_test = batch

                e_x = self.encoder(x, training=False)
                g_ex = self.generator(e_x, training=False)

                d_x, x_features = self.discriminator([x, e_x], training=False)
                d_gex, ex_features = self.discriminator([g_ex, e_x], training=False)

                # Losses: TODO: change the losses to accept the axis parameter
                # and reduce over axis [1,2,3] and preserve the batch dim
                # since we need the score per sample, not the average score
                # in the batch
                # d_loss = discriminator_loss(d_x, d_gex)
                # g_loss = adversarial_loss_fm(x_features, ex_features)

                g_score = tf.norm(
                    k.layers.Flatten()(residual_loss(x, g_ex)), axis=1, keepdims=False
                )
                d_score = tf.norm(
                    k.layers.Flatten()(x_features - ex_features), axis=1, keepdims=False
                )

                # Anomaly score should have a shape of (batch_size,)
                anomaly_scores = tf.linalg.normalize(
                    self._alpha * d_score + (1.0 - self._alpha) * g_score
                )

                # Update streaming auprc
                self._auprc.update_state(labels_test, anomaly_scores[0])

                # Save the model when AUPRC is the best
                current_auprc = self._auprc.result()

                if best_auprc < current_auprc:

                    tf.print("Best AUPRC on validation set: ", current_auprc)

                    # Replace the best
                    best_auprc = current_auprc

                    base_path = self._log_dir / "results" / "best"

                    self.generator.save_weights(
                        str(base_path / "generator"), overwrite=True
                    )

                    self.encoder.save_weights(
                        str(base_path / "encoder"), overwrite=True
                    )

                    with open(base_path / "auprc.json", "w") as fp:
                        json.dump(
                            {
                                "value": float(best_auprc),
                            },
                            fp,
                        )
            # Reset metrics or the data will keep accruing becoming an average of ALL the epochs
            self._reset_keras_metrics()

    # @tf.function
    def train_step(
        self,
        inputs,
    ):
        """Single training step."""
        x, _ = inputs
        noise = tf.random.normal(
            (tf.shape(x)[0], 1, 1, self._hps["latent_vector_size"])
        )
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction
            g_z = self.generator(noise, training=True)

            z_hat = self.encoder(x, training=True)
            xz_hat = self.generator(z_hat, training=True)

            d_x, x_features = self.discriminator([x, z_hat], training=True)
            d_g_z, g_z_features = self.discriminator([g_z, noise], training=True)

            # Losses
            d_loss = discriminator_loss(d_x, d_g_z)
            g_loss = adversarial_loss_fm(g_z_features, x_features)
            e_loss = encoder_loss(d_x)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        e_grads = tape.gradient(e_loss, self.encoder.trainable_variables)
        del tape

        self.optimizer_d.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )
        self.optimizer_g.apply_gradients(
            zip(g_grads, self.generator.trainable_variables)
        )
        self.optimizer_e.apply_gradients(zip(e_grads, self.encoder.trainable_variables))

        return (
            g_z,
            xz_hat,
            d_loss,
            g_loss,
            e_loss,
        )
