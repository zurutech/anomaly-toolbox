"""Collection of Trainers."""

from .anogan import AnoGAN, AnoGANMNIST
from .egbad import EGBAD
from .ganomaly import GANomaly

__ALL__ = ["AnoGAN", "AnoGANMNIST", "EGBAD", "GANomaly"]
