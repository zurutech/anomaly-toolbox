from typing import Dict, Type

from anomaly_toolbox.experiments.egbad import EGBADExperimentMNIST
from anomaly_toolbox.experiments.ganomaly import GANomalyExperimentMNIST
from anomaly_toolbox.experiments.interface import Experiment

AVAILABLE_EXPERIMENTS: Dict[str, Type[Experiment]] = {
    "ganomaly_mnist": GANomalyExperimentMNIST,
    "egbad_mnist": EGBADExperimentMNIST,
}
