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

"""All GANomaly experiments."""

from pathlib import Path
from typing import Dict

import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.hps import hparam_parser
from anomaly_toolbox.trainers.ganomaly import GANomaly


class GANomalyExperiment(Experiment):
    """
    GANomaly experiment.
    """

    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # Get the hyperparameters
        self._hps = hparam_parser(
            self._hparams_path,
            "ganomaly",
            list(self.hyperparameters().union(GANomaly.hyperparameters())),
        )

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.

        Args:
            hps: Dictionary with the parameters to use for the current run.
            log_dir: Where to store the tensorboard logs.
            dataset: The dataset to use for model training and evaluation.
        """
        print("Running GANomaly experiment...")

        summary_writer = tf.summary.create_file_writer(str(log_dir))
        new_size = (32, 32)

        # Create the dataset with the requested sizes (requested by the model architecture)
        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            class_label=hps["class_label"],
            batch_size=hps["batch_size"],
            new_size=new_size,
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )

        # Create the GANomaly trainer
        trainer = GANomaly(
            dataset=dataset,
            hps=hps,
            summary_writer=summary_writer,
            log_dir=log_dir,
        )

        # Train the GANomaly model
        trainer.train(
            epochs=hps["epochs"],
            adversarial_loss_weight=hps["adversarial_loss_weight"],
            contextual_loss_weight=hps["contextual_loss_weight"],
            enc_loss_weight=hps["enc_loss_weight"],
        )

        # Test on test dataset and put the results in the json file (the same file  used inside the
        # training for the model selection)
        trainer.test()
