"""All EGBAD experiments."""

from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.hps import grid_search
from anomaly_toolbox.experiments.interface import Experiment
from anomaly_toolbox.trainers import EGBAD

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
        hp.HParam("adversarial_loss_weight", hp.Discrete([1])),
        hp.HParam("contextual_loss_weight", hp.Discrete([50])),
        hp.HParam("enc_loss_weight", hp.Discrete([1])),
        hp.HParam("shuffle_buffer_size", hp.Discrete([10000])),
        hp.HParam("latent_vector_size", hp.Discrete([100])),
        hp.HParam("use_bce", hp.Discrete([True, False])),
    ]
    metrics: List[hp.Metric] = [
        hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
        hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
        hp.Metric("test_epoch_e_loss", display_name="Encoder Loss"),
    ]

    def experiment_run(self, hps: Dict, log_dir: str):
        """Perform a single run of the model."""
        summary_writer = tf.summary.create_file_writer(log_dir)
        trainer = EGBAD(
            input_dimension=self.input_dimension,
            filters=self.filters,
            hps=hps,
            summary_writer=summary_writer,
        )
        trainer.train_mnist(
            batch_size=hps["batch_size"],
            epoch=hps["epoch"],
            anomalous_label=hps["anomalous_label"],
            use_bce=hps["use_bce"],
            adversarial_loss_weight=hps["adversarial_loss_weight"],
            contextual_loss_weight=hps["contextual_loss_weight"],
            enc_loss_weight=hps["enc_loss_weight"],
        )

    def run(self):
        """Run the Experiment."""
        grid_search(
            self.experiment_run,
            hps=self.hps,
            metrics=self.metrics,
            log_dir=self.log_dir,
        )
