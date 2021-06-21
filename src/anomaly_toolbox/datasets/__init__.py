"""Datasets for anomaly detection module."""

from .corrupted_mnist import CorruptedMNIST
from .mnist import MNIST
from .surface_cracks import SurfaceCracks

__ALL__ = ["MNIST", "CorruptedMNIST", "SurfaceCracks"]
