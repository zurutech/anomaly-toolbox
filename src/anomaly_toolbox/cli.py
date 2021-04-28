# -*- coding: utf-8 -*-

"""Console script for anomaly_toolbox."""
import sys
from datetime import datetime

import click

from anomaly_toolbox.trainers import GANomaly


@click.command()
def main(args=None):
    """Console script for anomaly_toolbox."""
    click.echo(
        "Replace this message by putting your code into " "anomaly_toolbox.cli.main"
    )
    click.echo("See click documentation at http://click.pocoo.org/")

    hps = {
        "anomalous_label": 9,
        "learning_rate": 2e-3,
        "batch_size": 32,
        "epoch": 15,
        "adversarial_loss_weight": 1,
        "contextual_loss_weight": 50,
        "enc_loss_weight": 1,
        "shuffle_buffer_size": 10000,
    }
    id = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = "logs/train_data/" + id
    test_log_dir = "logs/test_data/" + id
    GANomaly(
        learning_rate=hps["learning_rate"],
        train_log_dir=train_log_dir,
        test_log_dir=test_log_dir,
    ).train_mnist(
        batch_size=hps["batch_size"],
        epoch=hps["epoch"],
        anomalous_label=hps["anomalous_label"],
        adversarial_loss_weight=hps["adversarial_loss_weight"],
        contextual_loss_weight=hps["contextual_loss_weight"],
        enc_loss_weight=hps["enc_loss_weight"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
