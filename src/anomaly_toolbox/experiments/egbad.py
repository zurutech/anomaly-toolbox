"""All EGBAD experiments."""

from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.experiments.interface import Experiment
from anomaly_toolbox.trainers import EGBAD
from anomaly_toolbox.datasets import MNIST
from anomaly_toolbox.hps import hparam_parser


__ALL__ = ["EGBADExperimentMNIST"]


class EGBADExperimentMNIST(Experiment):
    """
    EGBAD experiment on MNIST.
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

    input_dimension: Tuple[int, int, int] = (32, 32, 1)
    filters: int = 64

    metrics: List[hp.Metric] = [
        hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
        hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
        hp.Metric("test_epoch_e_loss", display_name="Encoder Loss"),
    ]

    def experiment_run(self, hps: Dict, log_dir: Path):
        """Perform a single run of the model."""
        summary_writer = tf.summary.create_file_writer(str(log_dir))

        # Create the dataset
        mnist_dataset = MNIST()
        mnist_dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            new_size=(32, 32),
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )

        trainer = EGBAD(
            dataset=mnist_dataset,
            input_dimension=self.input_dimension,
            filters=self.filters,
            hps=hps,
            summary_writer=summary_writer,
        )

        trainer.train_mnist(
            epoch=hps["epochs"],
        )
