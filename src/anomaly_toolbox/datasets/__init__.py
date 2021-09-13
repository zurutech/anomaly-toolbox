# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
