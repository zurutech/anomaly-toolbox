"""Trainer for the GANomaly model."""

from json import decoder
from typing import Tuple

import tensorflow as tf
import tensorflow.keras as keras

from anomaly_toolbox.models import GANomalyDiscriminator, GANomalyGenerator

__ALL__ = ["GANomaly"]


class GANomaly:
    """GANomaly Trainer."""

    input_dimension: Tuple[int, int, int] = (32, 32, 3)
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
