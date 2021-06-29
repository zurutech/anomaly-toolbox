"""Collection of Trainers."""

from .anogan import AnoGAN
from .descargan import DeScarGAN
from .egbad import EGBAD
from .ganomaly import GANomaly

__ALL__ = ["AnoGAN", "EGBAD", "GANomaly", "DeScarGAN"]
