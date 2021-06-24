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

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: dictionary with the parameters to use for the current run.
            log_dir: where to store the tensorboard logs.
            dataset: the datset to use for model trainign and evaluation.
        """
        summary_writer = tf.summary.create_file_writer(str(log_dir))

        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )
        trainer = AnoGAN(
            dataset=dataset,
            hps=hps,
            summary_writer=summary_writer,
        )
        trainer.train_mnist(
            epochs=hps["epochs"],
        )

        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            new_size=input_shape[:-1],
            output_range=(-1.0, 1.0),  # generator has a tanh in output
            shuffle_buffer_size=hps["shuffle_buffer_size"],
            cache=True,
        )
        summary_writer = tf.summary.create_file_writer(str(log_dir))
        trainer = AnoGAN(dataset, input_shape, hps, summary_writer)
        trainer.train(
            batch_size=hps["batch_size"],
            epochs=hps["epochs"],
            step_log_frequency=step_log_frequency,
        )
