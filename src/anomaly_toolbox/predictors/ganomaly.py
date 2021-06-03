from typing import Type

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

    def evaluate(self, dataset):
        anomaly_scores, labels = [], []
        for batch in dataset:
            a_score, y = self.evaluate_step(batch)
            anomaly_scores.extend(a_score.numpy())
            labels.extend(y.numpy())
        return anomaly_scores, labels

    @tf.function
    def evaluate_step(self, inputs):
        x, y = inputs
        z_x, x_hat, z_hat = self.generator(x)
        a_score = self.comput_anomaly_score(z_x, z_hat)
        return a_score, y

    @staticmethod
    def comput_anomaly_score(encoded_input, encoded_generated):
        anomaly_score = tf.reduce_mean(
            tf.math.squared_difference(encoded_input, encoded_generated), 1
        )
        return anomaly_score

    @staticmethod
    def weight_scores(a_scores):
        a_scores_weighted = (a_scores - np.min(a_scores)) / (
            np.max(a_scores) - np.min(a_scores)
        )
        return a_scores_weighted
