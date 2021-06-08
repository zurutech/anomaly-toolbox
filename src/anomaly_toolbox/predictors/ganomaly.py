from typing import Type, Tuple, List

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from anomaly_toolbox.models import GANomalyGenerator


class GANomalyPredictor:
    generator: keras.Model
    discriminator: keras.Model

    def load_from_savedmodel(self, generator_dir: str, discriminator_dir: str):
        self.generator: GANomalyGenerator = tf.keras.models.load_model(generator_dir)
        self.discriminator = tf.keras.models.load_model(discriminator_dir)

    def evaluate(self, dataset) -> Tuple[tf.Tensor, tf.Tensor]:
        """Evaluate for benchmark."""
        anomaly_scores, labels = [], []
        for batch in dataset:
            # a_score: [batch, 1, latent dimension]
            # y: [batch, 1]
            a_score, y = self.evaluate_step(batch)
            anomaly_scores.append(a_score)
            labels.append(y)
        anomaly_scores = tf.reshape(anomaly_scores, -1)
        labels = tf.reshape(labels, -1)
        tf.assert_equal(anomaly_scores.shape, labels.shape)
        return anomaly_scores, labels

    @tf.function
    def evaluate_step(self, inputs):
        # x: [batch, height, width, channels]
        # y: [batch, 1]
        x, y = inputs

        # z: [batch, 1, 1, latent dimension]
        # x_hat: [batch, height, width, channels]
        # z_hat: [batch, 1, 1, latent dimension]
        z, x_hat, z_hat = self.generator(x)

        # z: [batch, latent dimension]
        # z_hat: [batch, latent dimension]
        z, z_hat = tf.squeeze(z), tf.squeeze(z_hat)

        # a_score: [batch, 1]
        a_score = self.comput_anomaly_score(z, z_hat)

        return a_score, y

    @staticmethod
    @tf.function
    def predict(generator: keras.Model, x, return_score_only: bool = True):
        # x: [batch, height, width, channels]

        # z: [batch, 1, 1, latent dimension]
        # x_hat: [batch, height, width, channels]
        # z_hat: [batch, 1, 1, latent dimension]
        z, x_hat, z_hat = generator(x)

        # z: [batch, latent dimension]
        # z_hat: [batch, latent dimension]
        z, z_hat = tf.squeeze(z), tf.squeeze(z_hat)

        # a_score: [batch, 1]
        a_score = GANomalyPredictor.comput_anomaly_score(z, z_hat)
        if return_score_only:
            return a_score
        else:
            return x_hat, z_hat, z, a_score

    @staticmethod
    def comput_anomaly_score(encoded_input, encoded_generated):
        anomaly_score = tf.reduce_mean(
            tf.math.squared_difference(encoded_input, encoded_generated), axis=1
        )
        return anomaly_score

    @staticmethod
    def weight_scores(a_scores):
        a_scores_weighted = (a_scores - np.min(a_scores)) / (
            np.max(a_scores) - np.min(a_scores)
        )
        return a_scores_weighted
