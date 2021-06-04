import abc
from pathlib import Path
from typing import Dict, List


class Experiment(abc.ABC):
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir

    @abc.abstractproperty
    def hps(self):
        pass

    @abc.abstractproperty
    def metrics(self):
        pass

    @abc.abstractmethod
    def experiment_run(self, hps: Dict, log_dir: Path):
        pass

    @abc.abstractmethod
    def run(self):
        pass
