"""Datasets for anomaly detection module."""

from .corrupted_mnist import CorruptedMNIST
from .dataset import AnomalyDetectionDataset
from .mnist import MNIST
from .mvtecad import MVTecAD
from .surface_cracks import SurfaceCracks

__all__ = [
    "MNIST",
    "CorruptedMNIST",
    "SurfaceCracks",
    "AnomalyDetectionDataset",
    "MVTecAD",
]
__datasets__ = ["MNIST", "CorruptedMNIST", "SurfaceCracks", "MVTecAD"]
