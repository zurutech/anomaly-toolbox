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

"""Trainer for the BiGAN model used in EGBAD."""

import json
from pathlib import Path
from typing import Dict, Set, Union

import tensorflow as tf
import tensorflow.keras as k

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses.egbad import (
    AdversarialLoss,
    encoder_bce,
    generator_bce,
    residual_loss,
)
from anomaly_toolbox.models.egbad import Decoder, Discriminator, Encoder
from anomaly_toolbox.trainers.trainer import Trainer


class EGBAD(Trainer):
    """EGBAD Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize EGBAD-BiGAN Networks."""
        super().__init__(
            dataset=dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        # # Models # #
        n_channels = dataset.channels

        # Decoder (aka Generator)
        self.generator = Decoder(n_channels, self._hps["latent_vector_size"], 0.2)
        self.generator.summary()

        # Encoder
        self.encoder = Encoder(
            n_channels,
            self._hps["latent_vector_size"],
            0.2,
        )
        self.encoder.summary()

        # Discriminator
        self.discriminator = Discriminator(n_channels, self._hps["latent_vector_size"])

        # Instantiate and define with correct input shape
        fake_batch_size = (1, 28, 28, n_channels)
        fake_latent_vector = (1, self._hps["latent_vector_size"])
        self.generator.call(tf.zeros(fake_latent_vector))
        self.encoder.call(tf.zeros(fake_batch_size))
        self.discriminator([tf.zeros(fake_batch_size), tf.zeros(fake_latent_vector)])
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
        self._auc_rc = k.metrics.AUC(name="auc_rc", curve="PR", num_thresholds=500)
        self._auc_roc = k.metrics.AUC(name="auc_roc", curve="ROC", num_thresholds=500)

        self.keras_metrics = {
            metric.name: metric
            for metric in [
                self.epoch_d_loss_avg,
                self.epoch_g_loss_avg,
                self.epoch_e_loss_avg,
                self._auc_rc,
                self._auc_roc,
            ]
        }

        self._alpha = tf.constant(0.9)

        self._minmax = AdversarialLoss(from_logits=True)
        self._flatten = k.layers.Flatten()

    @staticmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        return {"learning_rate", "latent_vector_size"}

    def train(
        self,
        epochs: int,
        step_log_frequency: int = 100,
    ):
        best_auc_rc, best_auc_roc = -1, -1
        for epoch in tf.range(epochs):

            for batch in self._dataset.train_normal:
                x, _ = batch

                # Perform the train step
                g_z, g_ex, d_loss, g_loss, e_loss = self.train_step(x)

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
                        tf.summary.scalar(
                            "e_loss",
                            self.epoch_e_loss_avg.result(),
                            step=step,
                        )

                    tf.print(
                        "Step {:04d}: d_loss: {:.3f}, g_loss: {:.3f}, e_loss: {:.3f}, "
                        "lr: {:.5f}".format(
                            step,
                            self.epoch_d_loss_avg.result(),
                            self.epoch_g_loss_avg.result(),
                            self.epoch_e_loss_avg.result(),
                            learning_rate,
                        )
                    )

            # Epoch end
            tf.print(epoch, "Epoch completed")

            # Model selection at the end of every epoch on the validation set
            # 1. Compute the the anomaly score
            # 2. Use the AUC object to compute the AUC with different
            # thresholds values on the anomaly score.
            self._auc_rc.reset_state()
            self._auc_roc.reset_state()

            for batch in self._dataset.validation:
                x, labels_test = batch

                # Get the anomaly scores
                anomaly_scores = self._compute_anomaly_scores(
                    x, self.encoder, self.generator, self.discriminator
                )

                # Update streaming auc_rc
                self._auc_rc.update_state(labels_test, anomaly_scores)
                # Update streaming auc_roc
                self._auc_roc.update_state(labels_test, anomaly_scores)

            # Save the best models (AUC RC)
            current_auc_rc = self._auc_rc.result()
            if best_auc_rc < current_auc_rc:
                tf.print("Best AUPRC on validation set: ", current_auc_rc)

                # Replace the best
                best_auc_rc = current_auc_rc

                base_path = self._log_dir / "results" / "auc_rc"
                self.generator.save(str(base_path / "generator"), overwrite=True)
                self.encoder.save(str(base_path / "encoder"), overwrite=True)
                self.discriminator.save(
                    str(base_path / "discriminator"), overwrite=True
                )

                with open(base_path / "validation.json", "w") as fp:
                    json.dump(
                        {
                            "value": float(best_auc_rc),
                        },
                        fp,
                    )

            # Save the best models (AUC ROC)
            current_auc_roc = self._auc_rc.result()
            if best_auc_roc < current_auc_roc:
                tf.print("Best AUC ROC on validation set: ", current_auc_roc)

                # Replace the best
                best_auc_rc = current_auc_roc

                base_path = self._log_dir / "results" / "auc_roc"
                self.generator.save(str(base_path / "generator"), overwrite=True)
                self.encoder.save(str(base_path / "encoder"), overwrite=True)
                self.discriminator.save(
                    str(base_path / "discriminator"), overwrite=True
                )

                with open(base_path / "validation.json", "w") as fp:
                    json.dump(
                        {
                            "value": float(best_auc_rc),
                        },
                        fp,
                    )

            # Reset metrics or the data will keep accruing becoming an average of ALL the epochs
            self._reset_keras_metrics()

    @tf.function
    def train_step(
        self,
        x,
    ):
        """Single training step."""
        z = tf.random.normal((tf.shape(x)[0], self._hps["latent_vector_size"]))
        with tf.GradientTape(persistent=True) as tape:
            # Reconstruction
            g_z = self.generator.call(z, training=True)

            e_x = self.encoder.call(x, training=True)
            g_ex = self.generator.call(e_x, training=True)

            d_g_z, _ = self.discriminator(inputs=[g_z, z], training=True)
            d_x, _ = self.discriminator(inputs=[x, e_x], training=True)

            # Losses
            d_loss = self._minmax(d_x, d_g_z)

            # g_loss = self._bce(tf.ones_like(d_g_z), d_g_z)
            g_loss = generator_bce(d_g_z, from_logits=True)

            # e_loss = self._bce(tf.zeros_like(d_x), d_x) + residual_loss(x, g_ex)
            e_loss = encoder_bce(d_x, from_logits=True) + residual_loss(x, g_ex)

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
            g_ex,
            d_loss,
            g_loss,
            e_loss,
        )

    def test(self, base_path: Union[Path, None] = None):
        """Measure the performance on the test set.

        Args:
            base_path: the path to use for loading the models. If None, the default is used.
        """

        for metric in ["auc_rc", "auc_roc"]:
            if not base_path:
                base_path = self._log_dir / "results" / metric
            encoder_path = base_path / "encoder"
            generator_path = base_path / "generator"
            discriminator_path = base_path / "discriminator"

            # Load the best models to use as the model here
            encoder = tf.keras.models.load_model(encoder_path)
            encoder.summary()
            generator = tf.keras.models.load_model(generator_path)
            generator.summary()
            discriminator = tf.keras.models.load_model(discriminator_path)
            discriminator.summary()

            # Resetting the state of the AUPRC variable
            self._auc_rc.reset_states()

            # Test on the test dataset
            for batch in self._dataset.test:
                x, labels_test = batch

                anomaly_scores = self._compute_anomaly_scores(
                    x, encoder, generator, discriminator
                )

                self._auc_rc.update_state(labels_test, anomaly_scores)
                self._auc_roc.update_state(labels_test, anomaly_scores)

            auc_rc = self._auc_rc.result()
            auc_roc = self._auc_roc.result()

            tf.print("Best AUPRC on test set: ", auc_rc)
            tf.print("Best AUCROC on test set: ", auc_roc)

            base_path = self._log_dir / "results" / metric

            # Create the resulta
            result = {
                "auc_rc": {
                    "value": float(auc_rc),
                },
                "auc_roc": {
                    "value": float(auc_roc),
                },
            }

            # Write the file
            with open(base_path / "test.json", "w") as fp:
                json.dump(result, fp)

    def _compute_anomaly_scores(
        self, x: tf.Tensor, encoder: k.Model, generator: k.Model, discriminator: k.Model
    ) -> tf.Tensor:
        """
        Compute the anomaly scores as indicated in the EGBAD paper
        https://arxiv.org/abs/1802.06222.

        Args:
            x: The batch of data to use to calculate the anomaly scores.

        Returns:
            The anomaly scores on the input batch, [0, 1] normalized.

        """
        e_x = encoder.call(x, training=False)
        g_ex = generator.call(e_x, training=False)

        _, x_features = discriminator([x, e_x], training=False)
        _, ex_features = discriminator([g_ex, e_x], training=False)

        g_score = tf.norm(
            k.layers.Flatten()(residual_loss(x, g_ex)), axis=1, keepdims=False
        )

        # Remove unused (i.e., 1-shaped) axis by squeezing
        ex_features = tf.squeeze(ex_features)
        x_features = tf.squeeze(x_features)

        d_score = tf.norm(
            self._flatten(x_features - ex_features), axis=1, keepdims=False
        )

        # Anomaly score should have a shape of (batch_size,)
        normalized_anomaly_scores, _ = tf.linalg.normalize(
            self._alpha * d_score + (1.0 - self._alpha) * g_score
        )

        return normalized_anomaly_scores
