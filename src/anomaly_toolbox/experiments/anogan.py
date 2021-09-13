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

"""AnoGAN experiment suite."""

from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.hps import hparam_parser
from anomaly_toolbox.trainers import AnoGAN


class AnoGANExperiment(Experiment):
    """
    AnoGAN experiment.
    """

    # List of hyperparameters names (to get from JSON)
    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # Get the hyperparameters
        self._hps = hparam_parser(
            hparams_path,
            "anogan",
            list(self.hyperparameters().union(AnoGAN.hyperparameters())),
        )

    metrics: List[hp.Metric] = [
        hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
        hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
    ]

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: dictionary with the parameters to use for the current run.
            log_dir: where to store the tensorboard logs.
            dataset: the dataset to use for model training and evaluation.
        """
        print("Running AnoGAN experiment...")

        summary_writer = tf.summary.create_file_writer(str(log_dir))

        # Create and configure the dataset
        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            class_label=hps["class_label"],
            batch_size=hps["batch_size"],
            new_size=(28, 28),
            output_range=(-1.0, 1.0),  # generator has a tanh in output
            shuffle_buffer_size=hps["shuffle_buffer_size"],
            cache=True,
        )

        # Create the AnoGAN model
        trainer = AnoGAN(dataset, hps, summary_writer, log_dir)

        # Train the model
        trainer.train(
            epochs=hps["epochs"],
            step_log_frequency=hps["step_log_frequency"],
        )

        # Test, using the best model so far
        trainer.test()
