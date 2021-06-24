"""EGBAD experiments suite."""

from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.hps import hparam_parser
from anomaly_toolbox.trainers import EGBAD


class EGBADExperiment(Experiment):
    """
    EGBAD experiment.
    """

    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # List of hyperparameters names (to get from JSON)
        self._hyperparameters_names = [
            "anomalous_label",
            "epochs",
            "batch_size",
            "optimizer",
            "learning_rate",
            "shuffle_buffer_size",
            "latent_vector_size",
        ]

        # Get the hyperparameters
        self._hps = hparam_parser(
            self._hparams_path, "egbad", self._hyperparameters_names
        )

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: dictionary with the parameters to use for the current run.
            log_dir: where to store the tensorboard logs.
            dataset: the dataset to use for model training and evaluation.
        """

        summary_writer = tf.summary.create_file_writer(str(log_dir))

        # Create the dataset
        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            new_size=(32, 32),
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )

        trainer = EGBAD(
            dataset=dataset,
            input_dimension=self.input_dimension,
            filters=self.filters,
            hps=hps,
            summary_writer=summary_writer,
        )

        trainer.train_mnist(
            epoch=hps["epochs"],
        )
