import itertools
from typing import Dict, List, Tuple

import tensorflow as tf
from hps import convert_to_hps, grid_search
from tensorboard.plugins.hparams import api as hp
from trainers import GANomaly


class ExperimentMNIST:
    # TODO: Refactor this directly as a list as did with metrics
    experiment_setup = (
        {
            "anomalous_label": {"type": hp.Discrete, "value": [2]},
            "epoch": {"type": hp.Discrete, "value": [1, 5, 10, 15]},
            "batch_size": {"type": hp.Discrete, "value": [32]},
            "optimizer": {"type": hp.Discrete, "value": ["adam"]},
            "learning_rate": {"type": hp.Discrete, "value": [0.002, 0.001, 0.0005]},
            "adversarial_loss_weight": {"type": hp.Discrete, "value": [1]},
            "contextual_loss_weight": {"type": hp.Discrete, "value": [50]},
            "enc_loss_weight": {"type": hp.Discrete, "value": [1]},
            "shuffle_buffer_size": {"type": hp.Discrete, "value": [10000]},
            "latent_vector_size": {"type": hp.Discrete, "value": [50, 100]},
        },
        [
            hp.Metric("test_epoch_d_loss", display_name="Discriminator Loss"),
            hp.Metric("test_epoch_g_loss", display_name="Generator Loss"),
            hp.Metric("test_epoch_e_loss", display_name="Encoder Loss"),
        ],
    )

    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    @staticmethod
    def single_run(hps, log_dir: str):
        summary_writer = tf.summary.create_file_writer(log_dir)
        trainer = GANomaly(
            learning_rate=hps["learning_rate"],
            summary_writer=summary_writer,
            hps=hps,
        )
        trainer.train_mnist(
            batch_size=hps["batch_size"],
            epoch=hps["epoch"],
            anomalous_label=hps["anomalous_label"],
            adversarial_loss_weight=hps["adversarial_loss_weight"],
            contextual_loss_weight=hps["contextual_loss_weight"],
            enc_loss_weight=hps["enc_loss_weight"],
        )

    def run(self):
        grid_search(
            self.single_run,
            self.experiment_setup,
            self.log_dir,
        )
