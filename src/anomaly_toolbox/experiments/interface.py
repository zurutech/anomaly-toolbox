"""Experiment generic structure."""

import abc
from pathlib import Path
from typing import Dict, Callable


class Experiment(abc.ABC):
    """
    The Experiment class represent the basic class to be used by each experiment.
    """
    def __init__(self, hparams_file_path: Path, log_dir: Path) -> None:
        """
        Experiment ctor.

        Args:
            hparams_file_path: The path of the hparams JSON file.
            log_dir: The log dir to be used for the experiment.
        """
        self._hparams_path = hparams_file_path
        self._log_dir = log_dir
        self._hps = None

    @property
    @abc.abstractmethod
    def metrics(self):
        pass

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def hps(self):
        return self._hps

    @abc.abstractmethod
    def experiment_run(self, hps: Dict, log_dir: Path):
        pass

    def run(self, hparams_tuning: bool, hparams_func: Callable) -> None:
        """
        Run training. It can run with hyperparameters tuning (hparams_tuning == True) or it can
        run without tuning (hparams_tuning==False). The tuning function is passed as a callable.

        Args:
            hparams_tuning: True if you want to enable tuning, False otherwise.
            hparams_func: The tuning function (e.g., grid_search) to use to do hyperparameters
            tuning.
        """
        if hparams_tuning:
            hparams_func(
                self.experiment_run,
                hps=self.hps,
                metrics=self.metrics,
                log_dir=str(self.log_dir),
            )
        else:
            hps_run = {}
            for i, entry in enumerate(self.hps):
                hps_run[entry.name] = entry.domain.values[0]
            self.experiment_run(hps_run, self.log_dir)
