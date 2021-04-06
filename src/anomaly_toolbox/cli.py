# -*- coding: utf-8 -*-

"""Console script for anomaly_toolbox."""
import sys

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
        "batch_size": 64,
        "epoch": 10,
        "shuffle_buffer_size": 10000,
        "adversarial_loss_weight": 1,
        "contextual_loss_weight": 1,
        "enc_loss_weight": 1,
    }
    GANomaly(learning_rate=hps["learning_rate"]).train_mnist(
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
