"""All AnoGAN experiments."""

from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.experiments.interface import Experiment
from anomaly_toolbox.hps import grid_search
from anomaly_toolbox.trainers import AnoGAN, AnoGANMNIST

__ALL__ = ["AnoGANExperimentMNIST"]


class AnoGANExperimentMNIST(Experiment):
    # --- HPS ---
    hps: List[hp.HParam] = [
        hp.HParam("anomalous_label", hp.Discrete([2])),
        hp.HParam("epochs", hp.Discrete([1, 5, 10, 15])),
        hp.HParam("batch_size", hp.Discrete([32])),
        hp.HParam("optimizer", hp.Discrete(["adam"])),  # NOTE: Currently unused
        hp.HParam("learning_rate", hp.Discrete([0.002, 0.001, 0.0005])),
        hp.HParam("shuffle_buffer_size", hp.Discrete([10000])),
        hp.HParam("latent_vector_size", hp.Discrete([100])),
    ]
    metrics: List[hp.Metric] = [
        hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
        hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
    ]

    def experiment_run(self, hps: Dict, log_dir: Path):
        """Perform a single run of the model."""
        summary_writer = tf.summary.create_file_writer(str(log_dir))
        trainer = AnoGANMNIST(
            hps=hps,
            summary_writer=summary_writer,
        )
        trainer.train_mnist(
            batch_size=hps["batch_size"],
            epochs=hps["epochs"],
            anomalous_label=hps["anomalous_label"],
        )

    def run(self):
        """Run the Experiment."""
        grid_search(
            self.experiment_run,
            hps=self.hps,
            metrics=self.metrics,
            log_dir=str(self.log_dir),
        )
