"""All EGBAD experiments."""

from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.experiments.interface import Experiment
from anomaly_toolbox.hps import grid_search
from anomaly_toolbox.trainers import EGBAD
from anomaly_toolbox.datasets import MNIST


__ALL__ = ["EGBADExperimentMNIST"]


class EGBADExperimentMNIST(Experiment):
    input_dimension: Tuple[int, int, int] = (32, 32, 1)
    filters: int = 64
    # --- HPS ---
    hps: List[hp.HParam] = [
        hp.HParam("anomalous_label", hp.Discrete([2])),
        hp.HParam("epoch", hp.Discrete([1, 5, 10, 15])),
        hp.HParam("batch_size", hp.Discrete([32])),
        hp.HParam("optimizer", hp.Discrete(["adam"])),  # NOTE: Currently unused
        hp.HParam("learning_rate", hp.Discrete([0.002, 0.001, 0.0005])),
        hp.HParam("shuffle_buffer_size", hp.Discrete([10000])),
        hp.HParam("latent_vector_size", hp.Discrete([128])),
    ]
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
        mnist_dataset.assemble_datasets(
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
            epoch=hps["epoch"],
        )

    def run(self):
        """Run the Experiment."""
        grid_search(
            self.experiment_run,
            hps=self.hps,
            metrics=self.metrics,
            log_dir=str(self.log_dir),
        )
