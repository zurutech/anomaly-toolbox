"""All AnoGAN experiments."""

from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.experiments.interface import Experiment
from anomaly_toolbox.trainers import AnoGANMNIST
from anomaly_toolbox.datasets import MNIST
from anomaly_toolbox.hps import hparam_parser

__ALL__ = ["AnoGANExperimentMNIST"]


class AnoGANExperimentMNIST(Experiment):
    """
    AnoGAN experiment on MNIST.
    """

    # List of hyperparameters names (to get from JSON)
    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # List of hyperparameters names (to get from JSON)
        self._hparams_path = hparams_path
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
            self._hparams_path, "anogan", self._hyperparameters_names
        )

    metrics: List[hp.Metric] = [
        hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
        hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
    ]

    def experiment_run(self, hps: Dict, log_dir: Path):
        """Perform a single run of the model."""
        summary_writer = tf.summary.create_file_writer(str(log_dir))

        mnist_dataset = MNIST()
        mnist_dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )
        trainer = AnoGANMNIST(
            dataset=mnist_dataset,
            hps=hps,
            summary_writer=summary_writer,
        )
        trainer.train_mnist(
            epochs=hps["epochs"],
        )
