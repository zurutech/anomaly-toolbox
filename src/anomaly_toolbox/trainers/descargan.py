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

"""Trainer for the DeScarGAN model."""

import json
from pathlib import Path
from typing import Dict, Set, Tuple, Union

import tensorflow as tf
import tensorflow.keras as k

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.models.descargan import Discriminator, Generator
from anomaly_toolbox.trainers.trainer import Trainer


class DeScarGAN(Trainer):
    """DeScarGAN Trainer."""

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        hps: Dict,
        summary_writer: tf.summary.SummaryWriter,
        log_dir: Path,
    ):
        """Initialize DeScarGAN Trainer."""
        super().__init__(
            dataset, hps=hps, summary_writer=summary_writer, log_dir=log_dir
        )

        # Data info
        self._ill_label = dataset.anomalous_label
        self._healthy_label = dataset.normal_label

        # Models
        self.generator = Generator(
            ill_label=self._ill_label, n_channels=dataset.channels
        )
        self.discriminator = Discriminator(
            ill_label=self._ill_label, n_channels=dataset.channels
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
        self._g_lambda_classification = tf.constant(1.0)

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
        self.accuracy = k.metrics.BinaryAccuracy(name="binary_accuracy")
        self.keras_metrics = {
            metric.name: metric
            for metric in [self.epoch_d_loss_avg, self.epoch_g_loss_avg, self.accuracy]
        }
        # Outside of the keras_metrics because it's used to compute the running
        # mean and not as a metric
        self._mean = k.metrics.Mean()

        # Constants
        self._zero = tf.constant(0.0)
        self._zero_batch = tf.zeros((1, 1, 1, 1))

    @staticmethod
    def hyperparameters() -> Set[str]:
        """List of the hyperparameters name used by the trainer."""
        return {"learning_rate"}

    @staticmethod
    def clip_by_norm_handle_none(grad, clip_norm):
        """
        tape.compute_gradients returns None instead of a zero tensor when the
        gradient would be a zero tensor and tf.clip_by_* does not support
        a None value.
        So just don't pass None to it and preserve None values.
        Source: https://stackoverflow.com/a/39295309/2891324
        """
        if grad is None:
            return None
        return tf.clip_by_norm(grad, clip_norm=clip_norm)

    def _select_and_save(self, threshold: tf.Tensor):
        current_accuracy = self.accuracy.result()
        base_path = self._log_dir / "results" / "accuracy"
        self.generator.save(
            str(base_path / "generator"),
            overwrite=True,
            include_optimizer=False,
        )
        with open(base_path / "validation.json", "w") as fp:
            json.dump(
                {
                    "value": float(current_accuracy),
                    "threshold": float(threshold),
                },
                fp,
            )

    @tf.function
    def train(
        self,
        epochs: int,
        step_log_frequency: int = 100,
    ):
        """
        Train the DeScarGAN generator and discriminator.
        """

        step_log_frequency = tf.convert_to_tensor(step_log_frequency, dtype=tf.int64)
        epochs = tf.convert_to_tensor(epochs, dtype=tf.int32)
        best_accuracy = -1.0
        for epoch in tf.range(epochs):
            for batch in self._dataset.train:
                # Perform the train step
                d_loss, g_loss, reconstructions = self.train_step(batch)

                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)

                # step -1 because in train_step self.d_optimizer.iterations has been incremented
                if tf.math.equal(
                    tf.math.mod(self.d_optimizer.iterations - 1, step_log_frequency),
                    tf.constant(0, tf.int64),
                ):
                    x, y = batch

                    healthy_idx = tf.squeeze(tf.where(tf.equal(y, self._healthy_label)))
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
                    with self._summary_writer.as_default():
                        tf.summary.scalar(
                            "d_loss", d_loss, step=self.d_optimizer.iterations
                        )
                        tf.summary.scalar(
                            "g_loss", g_loss, step=self.d_optimizer.iterations
                        )

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
                            step=self.d_optimizer.iterations,
                        )
                        tf.summary.image(
                            "ill",
                            tf.concat(
                                [x_ill, x_hat_ill, tf.abs(x_ill - x_hat_ill)], axis=2
                            ),
                            step=self.d_optimizer.iterations,
                        )
                    tf.print(
                        "[",
                        epoch,
                        "] step: ",
                        self.d_optimizer.iterations,
                        ": d_loss: ",
                        d_loss,
                        ", g_loss: ",
                        g_loss,
                    )
            # Epoch end

            # Global classification (not pixel level)
            # 1. Find the reconstruction error on a sub-set of the validation set
            # that WON'T be used during the score computation.
            # Reason: https://stats.stackexchange.com/a/427468/91290
            #
            # "The error distribution on the training data is misleading since your
            # training error distribution is not identical to test error distribution,
            # due to inevitable over-fitting. Then, comparing training error
            # distribution with future data is unjust."
            #
            # 2. Use the threshold to classify the validation set (positive and negative)
            # 3. Compute the binary accuracy (we can use it since the dataset is perfectly balanced)
            self._mean.reset_state()
            for x, y in self._dataset.validation_normal:
                self._mean.update_state(
                    tf.reduce_mean(
                        tf.math.abs(self.generator((x, y), training=False) - x)
                    )
                )
            threshold = self._mean.result()
            tf.print(
                "Reconstruction error on normal validation set: ",
                threshold,
            )

            # reconstruction <= threshold, then is a normal data (label 0)
            for x, y in self._dataset.test_normal.concatenate(
                self._dataset.test_anomalous
            ):
                self.accuracy.update_state(
                    y_true=y,
                    y_pred=tf.cast(
                        # reconstruction > threshold, then is anomalous (label 1 = cast(True))
                        # invoke the generator always with the normal label, since that's
                        # what we suppose to receive in input (and the threshold has been found
                        # using data that comes only from the normal distribution)
                        tf.math.greater(
                            tf.reduce_mean(
                                tf.math.abs(
                                    self.generator(
                                        (
                                            x,
                                            tf.ones(tf.shape(x)[0], dtype=tf.int32)
                                            * self._dataset.normal_label,
                                        ),
                                        training=False,
                                    )
                                    - x
                                ),
                                axis=[1, 2, 3],
                            ),
                            threshold,
                        ),
                        tf.int32,
                    ),
                )
            current_accuracy = self.accuracy.result()
            tf.print("Binary accuracy on validation set: ", current_accuracy)

            if best_accuracy < current_accuracy:
                tf.py_function(self._select_and_save, [threshold], [])
                best_accuracy = current_accuracy

            with self._summary_writer.as_default():
                tf.summary.scalar(
                    "accuracy", current_accuracy, step=self.d_optimizer.iterations
                )

            # Reset the metrics at the end of every epoch
            self._reset_keras_metrics()

    def gradient_penalty(
        self, x: tf.Tensor, x_gen: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute gradient penalty: L2(grad - 1)^2.

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
        """
        Single training step.

        Args:
            inputs: a tuple  (x,y) containing the input samples (x) and their labels (y).
                    both x, and y, are batches.
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
        percentage_healthy = tf.math.divide_no_nan(tot_healthy, tot)
        percentage_ill = tf.math.divide_no_nan(tot_ill, tot)

        # Scalar labels used in the losses
        healthy_labels = tf.ones((tot_healthy,), dtype=tf.int32) * tf.squeeze(
            self._healthy_label
        )
        ill_labels = tf.ones((tot_ill,), dtype=tf.int32) * tf.squeeze(self._ill_label)

        # Train the discriminator
        with tf.GradientTape(persistent=True) as tape:
            # With real images - healthy
            if tf.not_equal(percentage_healthy, self._zero):
                (d_healthy, d_healthy_pred) = self.discriminator(
                    [x_healthy, healthy_labels],
                    training=True,
                )
                d_loss_real_healthy = -tf.reduce_mean(d_healthy) * percentage_healthy
                d_loss_classification_healthy = (
                    self._classification_loss(
                        y_true=healthy_labels, y_pred=d_healthy_pred
                    )
                    * percentage_healthy
                )
            else:
                d_loss_classification_healthy = self._zero
                d_loss_real_healthy = self._zero

            # With real images - ill
            if tf.not_equal(percentage_ill, self._zero):
                (d_ill, d_ill_pred) = self.discriminator(
                    [x_ill, ill_labels], training=True
                )
                d_loss_real_ill = -tf.reduce_mean(d_ill) * percentage_ill
                d_loss_classification_ill = (
                    self._classification_loss(y_true=ill_labels, y_pred=d_ill_pred)
                    * percentage_ill
                )
            else:
                d_loss_classification_ill = self._zero
                d_loss_real_ill = self._zero

            # Total loss on real images
            d_loss_real = d_loss_real_ill + d_loss_real_healthy
            d_loss_classification = (
                d_loss_classification_ill + d_loss_classification_healthy
            )

            # Generate fake images:
            # Add random noise to the input too
            noise_variance = tf.constant(0.05)
            if tf.not_equal(percentage_healthy, self._zero):
                x_healthy_noisy = (
                    x_healthy
                    + tf.random.uniform(tf.shape(x_healthy), dtype=tf.float32)
                    * noise_variance
                )
                x_fake_healthy = self.generator(
                    [x_healthy_noisy, healthy_labels], training=True
                )
                # Add noise to generated and real images - used for the losses
                x_fake_healthy_noisy = (
                    x_fake_healthy
                    + tf.random.uniform(tf.shape(x_fake_healthy), dtype=tf.float32)
                    * noise_variance
                )
                # Train with fake noisy images
                (d_on_fake_healthy, _) = self.discriminator(
                    [x_fake_healthy_noisy, healthy_labels], training=True
                )

                # Gradient penealty
                d_gradient_penalty_healty = self.gradient_penalty(
                    x_healthy_noisy,
                    x_fake_healthy_noisy,
                    healthy_labels,
                )
            else:
                d_on_fake_healthy = self._zero
                d_gradient_penalty_healty = self._zero
                x_fake_healthy = self._zero
                x_fake_healthy_noisy = self._zero
                x_healthy_noisy = self._zero

            if tf.not_equal(percentage_ill, self._zero):
                x_ill_noisy = (
                    x_ill
                    + tf.random.uniform(tf.shape(x_ill), dtype=tf.float32)
                    * noise_variance
                )
                x_fake_ill = self.generator([x_ill_noisy, ill_labels], training=True)

                # Add noise to generated and real images - used for the losses
                x_fake_ill_noisy = (
                    x_fake_ill
                    + tf.random.uniform(tf.shape(x_fake_ill), dtype=tf.float32)
                    * noise_variance
                )

                # Train with fake noisy images
                (d_on_fake_ill, _) = self.discriminator(
                    [x_fake_ill_noisy, ill_labels], training=True
                )

                # Gradient penalty
                d_gradient_penalty_ill = self.gradient_penalty(
                    x_ill_noisy, x_fake_ill_noisy, ill_labels
                )
            else:
                d_on_fake_ill = self._zero_batch
                d_gradient_penalty_ill = self._zero
                x_fake_ill = self._zero_batch
                x_fake_ill_noisy = self._zero_batch
                x_ill_noisy = self._zero_batch

            d_loss_fake = (
                tf.reduce_mean(d_on_fake_healthy) * percentage_healthy
                + tf.reduce_mean(d_on_fake_ill) * percentage_ill
            )

            # Gradient penalty to improve discriminator training stability
            d_loss_gp = d_gradient_penalty_healty + d_gradient_penalty_ill

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

                if tf.not_equal(percentage_healthy, self._zero):
                    g_classification_loss_healthy = self._classification_loss(
                        y_true=healthy_labels,
                        y_pred=tf.reduce_mean(d_on_fake_healthy, axis=[2, 3]),
                    )
                    g_identity_loss_healthy = self._reconstruction_error(
                        y_true=x_healthy, y_pred=x_fake_healthy
                    )
                    g_reconstruction_loss_healthy = self._reconstruction_error(
                        y_true=x_healthy_noisy, y_pred=x_fake_healthy
                    )
                else:
                    g_classification_loss_healthy = self._zero
                    g_identity_loss_healthy = self._zero
                    g_reconstruction_loss_healthy = self._zero

                if tf.not_equal(percentage_ill, self._zero):
                    g_classification_loss_ill = self._classification_loss(
                        y_true=ill_labels,
                        y_pred=tf.reduce_mean(d_on_fake_ill, axis=[2, 3]),
                    )
                    g_identity_loss_ill = self._reconstruction_error(
                        y_true=x_ill, y_pred=x_fake_ill
                    )
                    g_reconstruction_loss_ill = self._reconstruction_error(
                        y_true=x_ill_noisy, y_pred=x_fake_ill
                    )
                else:
                    g_classification_loss_ill = self._zero
                    g_identity_loss_ill = self._zero
                    g_reconstruction_loss_ill = self._zero

                g_classification_loss = (
                    g_classification_loss_ill + g_classification_loss_healthy
                )

                # Adversarial loss
                g_loss_fake = -tf.reduce_mean(d_on_fake_healthy) - tf.reduce_mean(
                    d_on_fake_ill
                )

                # Identity loss
                g_identity_loss = g_identity_loss_ill + g_identity_loss_healthy

                # Reconstruction loss
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
                g_loss = self._zero

        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Gradient clipping
        d_grads = [self.clip_by_norm_handle_none(g, clip_norm=10) for g in d_grads]
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
            # Gradient clipping
            g_grads = [self.clip_by_norm_handle_none(g, clip_norm=10) for g in g_grads]
            self.g_optimizer.apply_gradients(
                zip(g_grads, self.generator.trainable_variables)
            )
        del tape

        return d_loss, g_loss, x_hat

    def test(self, base_path: Union[Path, None] = None):
        """Measure the performance (only measured metric is accuracy) on the
        test set.

        Args:
            base_path: the path to use for loading the models. If None, the default is used.
        """

        if not base_path:
            base_path = self._log_dir / "results" / "accuracy"

        # Load the best model to use as the model here
        model_path = base_path / "generator"
        generator = tf.keras.models.load_model(model_path)
        generator.summary()

        self.accuracy.reset_state()

        # Get the threshold
        accuracy_path = base_path / "validation.json"
        with open(accuracy_path, "r") as fp:
            data = json.load(fp)
            threshold = data["threshold"]

        # reconstruction <= threshold => normal data (label 0)
        for x, y in self._dataset.test_normal.concatenate(self._dataset.test_anomalous):
            self.accuracy.update_state(
                y_true=y,
                y_pred=tf.cast(
                    # reconstruction > threshold => anomalous (label 1 = cast(True))
                    # invoke the generator always with the normal label, since that's
                    # what we suppose to receive in input (and the threshold has been found
                    # using data that comes only from the normal distribution)
                    tf.math.greater(
                        tf.reduce_mean(
                            tf.math.abs(
                                generator(
                                    (
                                        x,
                                        tf.ones(tf.shape(x)[0], dtype=tf.int32)
                                        * self._dataset.normal_label,
                                    ),
                                    training=False,
                                )
                                - x
                            ),
                            axis=[1, 2, 3],
                        ),
                        threshold,
                    ),
                    tf.int32,
                ),
            )

        current_accuracy = self.accuracy.result()
        tf.print("Binary accuracy on test set: ", current_accuracy)

        # Create the result
        result_path = self._log_dir / "results" / "accuracy" / "test.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        with open(result_path, "w") as fp:
            json.dump(
                {
                    "accuracy": {
                        "value": float(current_accuracy),
                        "threshold": float(threshold),
                    }
                },
                fp,
            )
