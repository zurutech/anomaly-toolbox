"""Trainer for the GANomaly model."""

from typing import Tuple

import tensorflow as tf
import tensorflow.keras as keras
from models import GANomalyDiscriminator, GANomalyGenerator

__ALL__ = ["AnoGAN"]


class AnoGAN(Trainer):
    """AnoGAN Trainer."""

    input_dimension: Tuple[int, int, int] = (64, 64, 3)
    filters: int = 64
    latent_dimension: int = 100

    def __init__(self):
        """Initialize GANomaly Networks."""
        discriminator = GANomalyDiscriminator(
            self.input_dimension,
            self.filters,
        )
        generator = GANomalyGenerator(
            self.input_dimension, self.filters, self.latent_dimension
        )

    def train(self):
        pass

    def test_phas(self):
        pass

    @tf.function()
    def step_fn(self):
        pass
