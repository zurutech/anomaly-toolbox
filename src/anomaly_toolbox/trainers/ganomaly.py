"""Trainer for the GANomaly model."""

from json import decoder
from typing import Tuple

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python import training

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

    def __init__(self, learning_rate):
        """Initialize GANomaly Networks."""
        # -----
        # MODELS
        # NOTE: In our TF1 code we seem (I am not sure) to reuse the same encoder both for
        # GE and for E, however reading the paper it seems to me that they are different
        self.discriminator = GANomalyDiscriminator(
            self.input_dimension,
            self.filters,
        )

        # -----
        self.generator = GANomalyGenerator(
            self.input_dimension, self.filters, self.latent_dimension
        )

        # -----
        self.encoder = GANomalyAssembler.assemble_encoder(
            self.input_dimension, self.filters, self.latent_dimension
        )

        # -----
        # OPTIMIZERS
        # TODO: Can I use for schedule for all?
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            10000,
            0.5,
            staircase=True,
            name="LR_Scheduler",
        )
        self.optimizer_ge = keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999
        )
        self.optimizer_d = keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.5, beta_2=0.999
        )

    def train(
        self,
        dataset: tf.data.Dataset,
        steps_per_epoch: int,
        epoch: int,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):
        train_iterator = iter(dataset)
        for epoch in range(epoch):
            epoch_g_loss_avg = tf.keras.metrics.Mean(name="epoch_generator_loss")
            epoch_d_loss_avg = tf.keras.metrics.Mean(name="epoch_discriminator_loss")
            epoch_e_loss_avg = tf.keras.metrics.Mean(name="epoch_encoder_loss")
            for step in range(steps_per_epoch):
                x_hat, d_loss, e_loss, g_loss = self.train_step(
                    train_iterator,
                    use_bce,
                    adversarial_loss_weight,
                    contextual_loss_weight,
                    enc_loss_weight,
                )
                epoch_g_loss_avg.update_state(g_loss)
                epoch_d_loss_avg.update_state(d_loss)
                epoch_e_loss_avg.update_state(e_loss)
                # TODO: Log images on TB
                # TODO: Log losses on TB
                # tf.summary.scalar("d_loss", d_loss)
                # tf.summary.scalar("g_loss", g_loss)
                # tf.summary.scalar("e_loss", e_loss)
                step = self.optimizer_d.iterations.numpy()
                # print(
                #     f"Step: {step} - d_loss: {d_loss.numpy()}, g_loss: {g_loss.numpy()}, e_loss: {e_loss.numpy()}"
                # )
                if step == steps_per_epoch:
                    break
            print(
                "Epoch {:03d}: d_loss: {:.3f}, g_loss: {:.3f}, e_loss: {:.3f},".format(
                    epoch,
                    epoch_d_loss_avg.result(),
                    epoch_g_loss_avg.result(),
                    epoch_e_loss_avg.result(),
                )
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
        )

    def step_fn(
        self,
        inputs,
        use_bce: bool,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):
        x, y = inputs
        with tf.GradientTape(persistent=True) as tape:
            # Discriminator on real data
            d_x, d_f_x = self.discriminator(x, training=True)

            # Reconstruction
            z, x_hat = self.generator(x, training=True)
            z_hat = self.encoder(x_hat, training=True)

            # Discriminator on x_hat
            d_x_hat, d_f_x_hat = self.discriminator(x, training=True)

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
            x_hat,
            d_loss,
            e_loss,
            g_loss,
        )

    # @tf.function
    def train_mnist(
        self,
        batch_size: int,
        epoch: int,
        anomalous_label: int,
        adversarial_loss_weight: float,
        contextual_loss_weight: float,
        enc_loss_weight: float,
    ):
        ds_builder = MNISTDataset()
        (
            ds_train,
            ds_train_anomalous,
            ds_test,
            ds_test_anomalous,
        ) = ds_builder.assemble_datasets(
            anomalous_label=anomalous_label, batch_size=batch_size
        )
        self.train(
            dataset=ds_train,
            # NOTE: steps_per_epoch = 60000 - 6000 filtered anomalous digits
            steps_per_epoch=54000 // batch_size,
            epoch=epoch,
            use_bce=False,
            adversarial_loss_weight=adversarial_loss_weight,
            contextual_loss_weight=contextual_loss_weight,
            enc_loss_weight=enc_loss_weight,
        )
