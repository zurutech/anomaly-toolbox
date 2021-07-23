"""Trainer for the GANomaly model."""

from typing import Dict, Set
from pathlib import Path
import json
import os

import tensorflow as tf
import tensorflow.keras as keras

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.losses.ganomaly import AdversarialLoss, generator_bce
from anomaly_toolbox.models.ganomaly import Discriminator, Encoder, Decoder
from anomaly_toolbox.trainers.trainer import Trainer


class GANomaly(Trainer):
    """GANomaly Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
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
        self._auprc = tf.keras.metrics.AUC(name="auprc", curve="PR", num_thresholds=500)
        self._training_keras_metrics = [
            self.epoch_d_loss_avg,
            self.epoch_g_loss_avg,
            self._auprc,
        ]

        self.keras_metrics = {
            metric.name: metric
            for metric in [
                self.epoch_d_loss_avg,
                self.epoch_g_loss_avg,
                self._auprc,
            ]
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
    ):
        best_auprc = -1

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
                            learning_rate,
                        )
                    )

            # Epoch end
            tf.print(epoch, "Epoch completed")

            # Model selection with AUPRC
            self._auprc.reset_state()
            for batch in self._dataset.validation:
                x, labels_test = batch

                anomaly_scores = self._compute_anomaly_scores(
                    x, self.encoder, self.generator
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

                self.generator.save(str(base_path / "generator"), overwrite=True)

                self.encoder.save(str(base_path / "encoder"), overwrite=True)

                self.discriminator.save(
                    str(base_path / "discriminator"), overwrite=True
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

    @tf.function
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
            g_z = self.generator(z, training=True)  # or False?

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

            l1_loss = self._mae(x, g_ex)  # Contextual loss
            e_loss = self._mse(e_x, e_gex)  # Encoder loss

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

        # NOTE: If d_loss = self._minmax(d_x_features, d_gex_features), dense layer would return
        # a warning. To suppress the warning "Gradients does not exist for variables" try using
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

    def test(self):

        base_path = self._log_dir / "results" / "best"
        encoder_path = base_path / "encoder"
        generator_path = base_path / "generator"

        # Load the best models to use as the model here
        encoder = tf.keras.models.load_model(encoder_path)
        encoder.summary()
        generator = tf.keras.models.load_model(generator_path)
        generator.summary()

        # Resetting the state of the AUPRC variable
        self._auprc.reset_states()

        # Test on the test dataset
        for batch in self._dataset.test:

            x, labels_test = batch

            # Get the anomaly scores
            anomaly_scores = self._compute_anomaly_scores(x, encoder, generator)

            # Update streaming auprc
            self._auprc.update_state(labels_test, anomaly_scores[0])

        # Get the current AUPRC value
        auprc = self._auprc.result()

        tf.print("Best AUPRC on test set: ", auprc)

        base_path = self._log_dir / "results" / "best"
        result_json_path = os.path.join(base_path, "auprc.json")

        # Update the file with the test results
        with open(result_json_path, "r") as file:
            data = json.load(file)

        # Append the result
        data["best_on_test_dataset"] = float(auprc)

        # Write the file
        with open(result_json_path, "w") as fp:
            json.dump(data, fp)

    def _compute_anomaly_scores(
        self, x: tf.Tensor, encoder: keras.Model, generator: keras.Model
    ) -> tf.Tensor:
        """
        Compute the anomaly scores as indicated in the GANomaly paper
        https://arxiv.org/abs/1805.06725.

        Args:
            x: The batch of data to use to calculate the anomaly scores.

        Returns:
            The anomaly scores on the input batch, [0, 1] normalized.

        """

        # Get the generator reconstruction of a decoded input data
        e_x = encoder(x, training=False)
        g_ex = generator(e_x, training=False)

        # Encode the generated g_ex
        e_gex = encoder(g_ex, training=False)

        # Get the anomaly scores
        anomaly_scores = tf.linalg.normalize(
            tf.norm(
                tf.keras.layers.Flatten()(tf.abs(e_x - e_gex)),
                axis=1,
                keepdims=False,
            )
        )

        return anomaly_scores
