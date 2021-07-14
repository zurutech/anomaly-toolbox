"""Collection of Trainers."""

from .anogan import AnoGAN
from .descargan import DeScarGAN
from .egbad import EGBAD
from .ganomaly import GANomaly

__all__ = ["AnoGAN", "EGBAD", "GANomaly", "DeScarGAN"]
