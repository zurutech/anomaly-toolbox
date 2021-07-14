"""Datasets for anomaly detection module."""

from .corrupted_mnist import CorruptedMNIST
from .mnist import MNIST
from .surface_cracks import SurfaceCracks
from .dataset import AnomalyDetectionDataset

__all__ = ["MNIST", "CorruptedMNIST", "SurfaceCracks", "AnomalyDetectionDataset"]
__datasets__ = ["MNIST", "CorruptedMNIST", "SurfaceCracks"]
