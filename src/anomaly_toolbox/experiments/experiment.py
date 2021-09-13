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

"""Experiment generic structure."""

import abc
from pathlib import Path
from typing import Callable, Dict, List, Set

from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class Experiment(abc.ABC):
    """
    The Experiment class represent the basic class to be used by each experiment.
    """

    def __init__(self, hparams_file_path: Path, log_dir: Path):
        """
        Experiment ctor.

        Args:
            hparams_file_path: The path of the hparams JSON file.
            log_dir: The log dir to use during the experiment.
        """
        self._hparams_path = hparams_file_path
        self._log_dir = log_dir
        self._hps = []

    @property
    def log_dir(self) -> Path:
        """The log dir to use during the experiment."""
        return self._log_dir

    @property
    def hps(self) -> List[hp.HParam]:
        """The list of the hyperparameters."""
        return self._hps

    @staticmethod
    def hyperparameters() -> Set[str]:
        """Common hyperparameters to all the experiments. These are the
        data-related hyperparameters."""
        return {
            "anomalous_label",
            "class_label",
            "epochs",
            "batch_size",
            "shuffle_buffer_size",
            "step_log_frequency",
        }

    @abc.abstractmethod
    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: Dictionary with the parameters to use for the current run.
            log_dir: Where to store the tensorboard logs.
            dataset: The datset to use for model training and evaluation.
        """
        raise NotImplementedError

    def run(
        self,
        hparams_tuning: bool,
        hparams_func: Callable,
        dataset: AnomalyDetectionDataset,
    ) -> None:
        """
        Run training. It can run with hyperparameters tuning (hparams_tuning==True) or it can
        run without tuning (hparams_tuning==False). The tuning function is passed as a callable.

        Args:
            hparams_tuning: True if you want to enable tuning, False otherwise.
            hparams_func: The tuning function (e.g., grid_search) to use to do hyperparameters
            tuning.
            dataset: The datset to use for model training and evaluation.
        """
        if hparams_tuning:
            hparams_func(
                self.experiment,
                hps=self.hps,
                log_dir=self.log_dir,
                dataset=dataset,
            )
        else:
            hps_run = {entry.name: entry.domain.values[0] for entry in self.hps}
            self.experiment(hps_run, self.log_dir, dataset)
