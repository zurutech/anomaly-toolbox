"""Official Experiments configuration."""

from anomaly_toolbox.experiments.anogan import AnoGANExperiment
from anomaly_toolbox.experiments.descargan import DeScarGANExperiment
from anomaly_toolbox.experiments.egbad import EGBADExperiment
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.experiments.ganomaly import GANomalyExperiment

__all__ = [
    "AnoGANExperiment",
    "DeScarGANExperiment",
    "EGBADExperiment",
    "GANomalyExperiment",
    "Experiment",
]

__experiments = [
    "AnoGANExperiment",
    "DeScarGANExperiment",
    "EGBADExperiment",
    "GANomalyExperiment",
]
