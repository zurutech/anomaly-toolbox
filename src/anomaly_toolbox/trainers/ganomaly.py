"""Trainer for the GANomaly model."""

from json import decoder
import tensorflow as tf
import tensorflow.keras as keras
from anomaly_toolbox.models import GANomaly
from typing import Tuple

__ALL__ = ["GANomalyTrainer"]


class GANomalyTrainer:
    encoder_input_dimension: Tuple[int, int, int] = (32, 32, 3)
    encoder_latent_dimension: int = 100

    def __init__(self):
        # Initialize GANomaly
        encoder = GANomaly.encoder(self.encoder_input_dimension)
        discriminator, discriminator_intermediate_layer = GANomaly.discriminator(
            self.encoder_input_dimension
        )
        decoder = GANomaly.decoder(input_dimension=self.encoder_latent_dimension)
