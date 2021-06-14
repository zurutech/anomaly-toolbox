import abc
from pathlib import Path
from typing import Dict, List


class Experiment(abc.ABC):
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir

    @property
    @abc.abstractmethod
    def hps(self):
        pass

    @property
    @abc.abstractmethod
    def metrics(self):
        pass

    @abc.abstractmethod
    def experiment_run(self, hps: Dict, log_dir: str):
        pass

    @abc.abstractmethod
    def run(self):
        pass
