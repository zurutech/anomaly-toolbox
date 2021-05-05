"""Trainer for the GANomaly model."""

from typing import Optional, Tuple

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets import MNISTDataset
from anomaly_toolbox.losses import ganomaly as losses
from anomaly_toolbox.models import (
    GANomalyAssembler,
    GANomalyDiscriminator,
    GANomalyGenerator,
)

__ALL__ = ["GANomaly"]


class GANomaly:
    """GANomaly Trainer."""

    input_dimension: Tuple[int, int, int] = (32, 32, 1)
    filters: int = 64
    latent_dimension: int = 100
    ds_train: tf.data.Dataset
    ds_train_anomalous: tf.data.Dataset
    ds_test: tf.data.Dataset
    ds_test_anomalous: tf.data.Dataset

    def __init__(self, learning_rate: float, summary_writer, hps):
        """Initialize GANomaly Networks."""
        print("Initializing GANomaly")
        # |--------|
        # | MODELS |
        # |--------|
        # NOTE: In our TF1 code we seem (I am not sure) to reuse the same encoder both for
        # GE and for E, however reading the paper it seems to me that they are different
        self.discriminator = GANomalyDiscriminator(
            self.input_dimension,
            self.filters,
        )
        self.generator = GANomalyGenerator(
            self.input_dimension, self.filters, self.latent_dimension
        )
        self.encoder = GANomalyAssembler.assemble_encoder(
            self.input_dimension, self.filters, self.latent_dimension
        )

        # |------------|
        # | OPTIMIZERS |
        # |------------|
        self.optimizer_ge = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.5, beta_2=0.999
        )

        # |---------|
        # | Metrics |
        # |---------|
        # Training Metrics
        self.epoch_d_loss_avg = tf.keras.metrics.Mean(name="epoch_discriminator_loss")
        self.epoch_g_loss_avg = tf.keras.metrics.Mean(name="epoch_generator_loss")
        self.epoch_e_loss_avg = tf.keras.metrics.Mean(name="epoch_encoder_loss")
        _training_metrics = [
            self.epoch_d_loss_avg,
            self.epoch_g_loss_avg,
            self.epoch_e_loss_avg,
        ]
        # Test Metrics
        self.test_d_loss_avg = tf.keras.metrics.Mean(name="test_discriminator_loss")
        self.test_g_loss_avg = tf.keras.metrics.Mean(name="test_generator_loss")
        self.test_e_loss_avg = tf.keras.metrics.Mean(name="test_encoder_loss")
        _test_metrics = [
            self.test_d_loss_avg,
            self.test_g_loss_avg,
            self.test_e_loss_avg,
        ]
        self._metrics = _training_metrics + _test_metrics

        # |---------|
        # | LOGGING |
        # |---------|
        self.summary_writer = summary_writer
        self.hps = hps

    def train(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        steps_per_epoch: int,
        epoch: int,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        step_log_frequency: int = 100,
        test_dataset: Optional[tf.data.Dataset] = None,
    ):
        train_iterator = iter(dataset)
        for epoch in range(epoch):
            training_data, training_reconstructions = [], []
            for _ in range(steps_per_epoch):
                # -----
                # Perform the train step
                (x, x_hat, d_loss, g_loss, e_loss,) = self.train_step(
                    train_iterator,
                    use_bce,
                    adversarial_loss_weight,
                    contextual_loss_weight,
                    enc_loss_weight,
                )
                # -----
                # Update the losses metrics
                self.epoch_d_loss_avg.update_state(d_loss)
                self.epoch_g_loss_avg.update_state(g_loss)
                self.epoch_e_loss_avg.update_state(e_loss)
                step = self.optimizer_d.iterations.numpy()
                learning_rate = self.optimizer_ge.learning_rate.numpy()
                # -----
                # Save the input images and their reconstructions for later use
                training_data.append(x)
                training_reconstructions.append(x_hat)
                # -----
                if step % step_log_frequency == 0:
                    with self.summary_writer.as_default():
                        tf.summary.scalar("learning_rate", learning_rate, step=step)
                    print(
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
                input_data=training_data[0][:batch_size],
                reconstructions=training_reconstructions[0][:batch_size],
                summary_writer=self.summary_writer,
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
                    use_bce=use_bce,
                    adversarial_loss_weight=adversarial_loss_weight,
                    contextual_loss_weight=contextual_loss_weight,
                    enc_loss_weight=enc_loss_weight,
                    batch_size=batch_size,
                    epoch=epoch,
                    step=step,
                )
            # Reset metrics or the data will keep accruing becoming an average of ALL the epcohs
            self._reset_metrics()

    def test_phase(
        self,
        test_dataset,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        batch_size: int,
        step: int,
        epoch: int,
        log: bool = True,
    ) -> Tuple:
        """Perform the test pass on a given test_dataset."""
        test_iterator = iter(test_dataset)
        testing_data, testing_reconstructions = [], []
        for input_data in test_iterator:
            (test_x, test_x_hat, test_d_loss, test_g_loss, test_e_loss) = self.step_fn(
                input_data,
                use_bce,
                adversarial_loss_weight,
                contextual_loss_weight,
                enc_loss_weight,
                training=False,
            )
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
                summary_writer=self.summary_writer,
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
    def train_step(
        self,
        train_iterator,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):
        return self.step_fn(
            next(train_iterator),
            use_bce=use_bce,
            adversarial_loss_weight=adversarial_loss_weight,
            contextual_loss_weight=contextual_loss_weight,
            enc_loss_weight=enc_loss_weight,
            training=True,
        )

    def step_fn(
        self,
        inputs,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        training: bool = True,
    ):
        x, y = inputs
        with tf.GradientTape(persistent=True) as tape:
            # Discriminator on real data
            d_x, d_f_x = self.discriminator(x, training=training)

            # Reconstruction
            z, x_hat = self.generator(x, training=training)
            z_hat = self.encoder(x_hat, training=training)

            # Discriminator on x_hat
            d_x_hat, d_f_x_hat = self.discriminator(x, training=training)

            # d loss
            d_loss = losses.discriminator_loss(d_x, d_x_hat)

            # g loss
            if use_bce:
                adversarial_loss = losses.adversarial_loss_bce(d_x_hat)
            else:
                adversarial_loss = losses.adversarial_loss_fm(d_f_x, d_f_x_hat)

            l1_loss = keras.losses.MeanAbsoluteError()(x, x_hat)
            e_loss = keras.losses.MeanSquaredError()(z, z_hat)  # l2_loss

            g_loss = (
                adversarial_loss_weight * adversarial_loss
                + contextual_loss_weight * l1_loss
                + enc_loss_weight * e_loss
            )

            # Loss Regularizers are computed manually
            regularizer = tf.keras.regularizers.l2()
            d_loss = regularizer(d_loss)
            e_loss = regularizer(e_loss)
            g_loss = regularizer(g_loss)

        if training:
            e_grads = tape.gradient(e_loss, self.encoder.trainable_variables)
            g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
            d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)

            self.optimizer_ge.apply_gradients(
                zip(e_grads, self.encoder.trainable_variables)
            )
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

    def train_mnist(
        self,
        batch_size: int,
        epoch: int,
        anomalous_label: int,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
        use_bce: bool,
    ) -> None:
        """
        Train GANomaly on MNIST dataset with one abnormal class.

        Args:
            batch_size:
            epoch:
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
            anomalous_label=anomalous_label, batch_size=batch_size
        )
        self.train(
            dataset=self.ds_train,
            # NOTE: steps_per_epoch = 60000 - 6000 filtered anomalous digits
            steps_per_epoch=54000 // batch_size,
            batch_size=batch_size,
            epoch=epoch,
            use_bce=use_bce,
            adversarial_loss_weight=adversarial_loss_weight,
            contextual_loss_weight=contextual_loss_weight,
            enc_loss_weight=enc_loss_weight,
            test_dataset=self.ds_test,
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
            hp.hparams(self.hps)
            # -----
            # Logging scalars
            # -----
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
            # -----
            # Logging images
            # -----
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

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset_states()
