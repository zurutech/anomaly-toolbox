"""Official Experiments configuration."""

from typing import Dict, Type

from anomaly_toolbox.experiments.anogan import AnoGANExperimentMNIST
from anomaly_toolbox.experiments.egbad import EGBADExperimentMNIST
from anomaly_toolbox.experiments.ganomaly import GANomalyExperimentMNIST
from anomaly_toolbox.experiments.interface import Experiment

AVAILABLE_EXPERIMENTS: Dict[str, Type[Experiment]] = {
    "anogan_mnist": AnoGANExperimentMNIST,
    "ganomaly_mnist": GANomalyExperimentMNIST,
    "egbad_mnist": EGBADExperimentMNIST,
}
