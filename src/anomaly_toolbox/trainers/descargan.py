"""Trainer for the DeScarGAN model."""

from typing import Dict, Tuple

import tensorflow as tf
import tensorflow.keras as k
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets import MNIST
from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.models.descargan import Discriminator, Generator
from anomaly_toolbox.trainers.trainer import Trainer


class DeScarGAN(Trainer):
    """DeScarGAN Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        input_shape: Tuple[int, int, int],
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
    ):
        """Initialize DeScarGAN Trainer."""
        super().__init__(dataset, hps=hps, summary_writer=summary_writer)

        # Data info
        self._ill_label = dataset.anomalous_label
        self._healthy_label = dataset.normal_label

        # Models
        self.generator = Generator(
            ill_label=self._ill_label, n_channels=input_shape[-1]
        )
        self.discriminator = Discriminator(
            ill_label=self._ill_label, n_channels=input_shape[-1]
        )

        # Optimizers
        self.g_optimizer = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )
        self.d_optimizer = k.optimizers.Adam(
            learning_rate=hps["learning_rate"], beta_1=0.5, beta_2=0.999
        )

        # Parameters
        self._generator_training_steps = tf.constant(5, dtype=tf.int64)
        self._g_lambda_identity = tf.constant(50.0)
        self._g_lambda_reconstruction = tf.constant(50.0)
        self._g_lambda_fake = tf.constant(1.0)
        self._g_lambda_classification = tf.constant(5.0)

        self._d_lambda_gradient_penalty = tf.constant(10.0)
        self._d_lambda_fake = tf.constant(20.0)
        self._d_lambda_real = tf.constant(20.0)
        self._d_lambda_classification = tf.constant(5.0)

        # Losses
        self._classification_loss = k.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self._reconstruction_error = k.losses.MeanSquaredError()

        # Training Metrics
        self.epoch_d_loss_avg = k.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_g_loss_avg = k.metrics.Mean(name="epoch_generator_loss")
        self._keras_metrics = [self.epoch_d_loss_avg, self.epoch_g_loss_avg]

    def _validate_models(self, input_shape: Tuple[int, int, int]):
        fake_batch_size = (1,) + input_shape
        inputs = [tf.zeros(fake_batch_size), tf.zeros((1,))]
        self.generator(inputs)
        self.discriminator(inputs)

        self.generator.summary()
        self.discriminator.summary()

    @tf.function
    def train(
        self,
        batch_size: int,
        epochs: int,
        step_log_frequency: int = 100,
    ):
        """Train the DeScarGAN generator and discriminator."""

        step_log_frequency = tf.convert_to_tensor(step_log_frequency, dtype=tf.int64)
        epochs = tf.convert_to_tensor(epochs, dtype=tf.int32)
        batch_size = tf.convert_to_tensor(batch_size, dtype=tf.int32)

        for epoch in tf.range(epochs):
            for batch in self._dataset.train:
                # Perform the train step
                d_loss, g_loss, reconstructions = self.train_step(batch)

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                step = self.d_optimizer.iterations

                if tf.math.equal(
                    tf.math.mod(step, step_log_frequency),
                    tf.constant(0, tf.int64),
                ):
                    with self._summary_writer.as_default():
                        x, y = batch

                        healthy_idx = tf.squeeze(
                            tf.where(tf.equal(y, self._healthy_label))
                        )
                        x_healthy = tf.gather(x, healthy_idx)
                        x_hat_healthy = tf.gather(reconstructions, healthy_idx)
                        if tf.equal(tf.rank(x_healthy), tf.constant(3)):
                            x_healthy = tf.expand_dims(x_healthy, axis=0)
                            x_hat_healthy = tf.expand_dims(x_hat_healthy, axis=0)

                        ill_idx = tf.squeeze(tf.where(tf.equal(y, self._ill_label)))
                        x_ill = tf.gather(x, ill_idx)
                        x_hat_ill = tf.gather(reconstructions, ill_idx)
                        if tf.equal(tf.rank(x_ill), tf.constant(3)):
                            x_ill = tf.expand_dims(x_ill, axis=0)
                            x_hat_ill = tf.expand_dims(x_hat_ill, axis=0)

                        tf.summary.scalar("d_loss", d_loss, step=step)
                        tf.summary.scalar("g_loss", g_loss, step=step)

                        tf.summary.image(
                            "healthy",
                            tf.concat(
                                [
                                    x_healthy,
                                    x_hat_healthy,
                                    tf.abs(x_healthy - x_hat_healthy),
                                ],
                                axis=2,
                            ),
                            step=step,
                        )
                        tf.summary.image(
                            "ill",
                            tf.concat(
                                [x_ill, x_hat_ill, tf.abs(x_ill - x_hat_ill)], axis=2
                            ),
                            step=step,
                        )
                    tf.print(
                        "[",
                        epoch,
                        "] step: ",
                        step,
                        ": d_loss: ",
                        d_loss,
                        ", g_loss: ",
                        g_loss,
                    )

            # Reset the metrics at the end of every epoch
            self._reset_keras_metrics()

    def gradient_penalty(
        self, x: tf.Tensor, x_gen: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute gradient penalty: L2(grad - 1)^2.
        Args:
            x: input batch
            x_gen: generated images
            labels: labels associated with x (and thus with x_gen)
        Returns:
            penalty on discriminator gradient
        """

        epsilon = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], 0.0, 1.0)
        x_hat = epsilon * x + (1 - epsilon) * x_gen
        labels = tf.cast(labels, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch([x_hat, labels])
            d_hat = self.discriminator([x_hat, labels], training=True)
        gradients = tape.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    @tf.function
    def train_step(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
    ):
        """Single training step.
        Args:
            inputs: a tuple  (x,y) containing the input samples (x) and their labels (y).
                    bothg x, and y, are batches.
                    If x is a batch of images the input shape is (batch_size, h, w, d).
                    The shape of y is always (batch_size,).
        Returns:
            d_loss, g_loss, x_hat, where

            d_loss: discriminator loss
            g_loss: generator loss
            x_hat: a tensor with the same shape of inputs[0] containing the reconstructions.
        """
        x, y = inputs

        # Generate all the reconstruction for the current batch
        # outside of the tape since we need this only for logging
        x_hat = self.generator(inputs, training=True)

        # All the gathering of the inputs can be done outside of the tape
        # no need to track these operations for computing the gradient (save memory)
        x_healthy = tf.gather(x, tf.squeeze(tf.where(tf.equal(y, self._healthy_label))))
        if tf.equal(tf.rank(x_healthy), tf.constant(3)):
            x_healthy = tf.expand_dims(x_healthy, axis=0)
        x_ill = tf.gather(x, tf.squeeze(tf.where(tf.equal(y, self._ill_label))))
        if tf.equal(tf.rank(x_ill), tf.constant(3)):
            x_ill = tf.expand_dims(x_ill, axis=0)

        # Count # healthy and # ill in batch
        tot_healthy = tf.cast(tf.shape(x_healthy)[0], tf.float32)
        tot_ill = tf.cast(tf.shape(x_ill)[0], tf.float32)
        tot = tf.cast(tf.shape(x)[0], tf.float32)
        percentage_healthy = tf.math.divide_no_nan(tot, tot_healthy)
        percentage_ill = tf.math.divide_no_nan(tot, tot_ill)

        # Scalar labels used in the losses
        healthy_labels = tf.ones((tot_healthy,), dtype=tf.int32) * tf.squeeze(
            self._healthy_label
        )
        ill_labels = tf.ones((tot_ill,), dtype=tf.int32) * tf.squeeze(self._ill_label)

        # Train the discriminator
        with tf.GradientTape(persistent=True) as tape:
            # With real images - healthy
            (d_healthy, d_healthy_pred) = self.discriminator(
                [x_healthy, healthy_labels],
                training=True,
            )
            d_loss_real_healthy = -tf.reduce_mean(d_healthy) * percentage_healthy
            d_loss_classification_healthy = (
                self._classification_loss(y_true=healthy_labels, y_pred=d_healthy_pred)
                * percentage_healthy
            )

            # With real images - ill
            (d_ill, d_ill_pred) = self.discriminator([x_ill, ill_labels], training=True)
            d_loss_real_ill = -tf.reduce_mean(d_ill) * percentage_ill
            d_loss_classification_ill = (
                self._classification_loss(y_true=ill_labels, y_pred=d_ill_pred)
                * percentage_ill
            )

            # Total loss on real images
            d_loss_real = d_loss_real_ill + d_loss_real_healthy
            d_loss_classification = (
                d_loss_classification_ill + d_loss_classification_healthy
            )

            # Generate fake images:
            # Add random noise to the input too

            noise_variance = tf.constant(0.05)
            x_healthy_noisy = (
                x_healthy
                + tf.random.uniform(tf.shape(x_healthy), dtype=tf.float32)
                * noise_variance
            )
            x_ill_noisy = (
                x_ill
                + tf.random.uniform(tf.shape(x_ill), dtype=tf.float32) * noise_variance
            )

            x_fake_healthy = self.generator(
                [x_healthy_noisy, healthy_labels], training=True
            )

            x_fake_ill = self.generator([x_ill_noisy, ill_labels], training=True)

            # Add noise to generated and real images - used for the losses
            x_fake_ill_noisy = (
                x_fake_ill
                + tf.random.uniform(tf.shape(x_fake_ill), dtype=tf.float32)
                * noise_variance
            )
            x_fake_healthy_noisy = (
                x_fake_healthy
                + tf.random.uniform(tf.shape(x_fake_healthy), dtype=tf.float32)
                * noise_variance
            )

            # Train with fake noiosy images
            (d_on_fake_healthy, _) = self.discriminator(
                [x_fake_healthy_noisy, healthy_labels], training=True
            )
            (d_on_fake_ill, _) = self.discriminator(
                [x_fake_ill_noisy, ill_labels], training=True
            )

            d_loss_fake = (
                tf.reduce_mean(d_on_fake_healthy) * percentage_healthy
                + tf.reduce_mean(d_on_fake_ill) * percentage_ill
            )

            # Gradient penalty to improve discriminator training stability
            d_loss_gp = (
                self.gradient_penalty(
                    x_healthy_noisy,
                    x_fake_healthy_noisy,
                    healthy_labels,
                )
                + self.gradient_penalty(x_ill_noisy, x_fake_ill_noisy, ill_labels)
            )

            # Sum all the losses and compute the discriminator loss
            d_loss = (
                self._d_lambda_real * d_loss_real
                + self._d_lambda_fake * d_loss_fake
                + self._d_lambda_classification * d_loss_classification
                + self._d_lambda_gradient_penalty * d_loss_gp
            )

            # Train the Generator ever self._generator_training_steps performed by the discriminator
            if tf.equal(
                tf.math.mod(
                    self.d_optimizer.iterations, self._generator_training_steps
                ),
                0,
            ):
                # D output reduction is needed because the output is batch_size, w, h, D
                g_classification_loss_healthy = self._classification_loss(
                    y_true=healthy_labels,
                    y_pred=tf.reduce_mean(d_on_fake_healthy, axis=[2, 3]),
                )
                g_classification_loss_ill = self._classification_loss(
                    y_true=ill_labels, y_pred=tf.reduce_mean(d_on_fake_ill, axis=[2, 3])
                )

                g_classification_loss = (
                    g_classification_loss_ill + g_classification_loss_healthy
                )

                # Adversarial loss
                g_loss_fake = -tf.reduce_mean(d_on_fake_healthy) - tf.reduce_mean(
                    d_on_fake_ill
                )

                # Identity loss
                g_identity_loss_healthy = self._reconstruction_error(
                    y_true=x_healthy, y_pred=x_fake_healthy
                )
                g_identity_loss_ill = self._reconstruction_error(
                    y_true=x_ill, y_pred=x_fake_ill
                )

                g_identity_loss = g_identity_loss_ill + g_identity_loss_healthy

                # Reconstruction loss
                g_reconstruction_loss_healthy = self._reconstruction_error(
                    y_true=x_healthy_noisy, y_pred=x_fake_healthy
                )
                g_reconstruction_loss_ill = self._reconstruction_error(
                    y_true=x_ill_noisy, y_pred=x_fake_ill
                )

                g_reconstruction_loss = (
                    g_reconstruction_loss_healthy + g_reconstruction_loss_ill
                )

                # Total generator loss
                g_loss = (
                    self._g_lambda_fake * g_loss_fake
                    + self._g_lambda_reconstruction * g_reconstruction_loss
                    + self._g_lambda_identity * g_identity_loss
                    + self._g_lambda_classification * g_classification_loss
                )
            else:
                g_loss = tf.constant(0.0)

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables)
        )

        if tf.equal(
            tf.cast(
                tf.math.mod(
                    self.d_optimizer.iterations - 1,
                    # -1 because at the previous line with d_opt.apply_gradients
                    # the counter increased
                    self._generator_training_steps,
                ),
                tf.int32,
            ),
            tf.constant(0, dtype=tf.int32),
        ):
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
        del tape

        return d_loss, g_loss, x_hat


if __name__ == "__main__":
    anomalous_label = 2
    epochs = 5
    batch_size = 50
    input_shape = (64, 64, 1)
    step_log_frequency = 50
    hps = {"learning_rate": hp.HParam("learning_rate", hp.Discrete([2e-5]))}

    log_dir = "cane"
    summary_writer = tf.summary.create_file_writer(log_dir)

    mnist_dataset = MNIST()

    mnist_dataset.configure(
        anomalous_label=anomalous_label,
        batch_size=batch_size,
        new_size=input_shape[:-1],
    )
    for lr in hps["learning_rate"].domain.values:
        hparams = {"learning_rate": lr}
        trainer = DeScarGAN(mnist_dataset, input_shape, hparams, summary_writer)
        trainer.train(
            batch_size=batch_size,
            epochs=epochs,
            step_log_frequency=step_log_frequency,
        )
        trainer.discriminator.save(log_dir + "/discriminator")
        trainer.generator.save(log_dir + "/generator")
