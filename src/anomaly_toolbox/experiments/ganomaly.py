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

        summary_writer = tf.summary.create_file_writer(str(log_dir))
        new_size = (32, 32)

        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            new_size=new_size,
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )
        trainer = GANomaly(
            dataset=dataset,
            # input_dimension=(new_size[0], new_size[1], dataset.channels),
            hps=hps,
            summary_writer=summary_writer,
            log_dir=log_dir,
        )
        trainer.train(
            epochs=hps["epochs"],
            adversarial_loss_weight=hps["adversarial_loss_weight"],
            contextual_loss_weight=hps["contextual_loss_weight"],
            enc_loss_weight=hps["enc_loss_weight"],
        )
