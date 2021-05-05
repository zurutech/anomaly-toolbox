# -*- coding: utf-8 -*-

"""Console script for anomaly_toolbox."""
import logging
import sys
from datetime import datetime

import click
import tensorflow as tf

from anomaly_toolbox.trainers import GANomaly
import anomaly_toolbox.experiments.ganomaly as ganomaly_experiments


@click.command()
def main(args=None):
    """Console script for anomaly_toolbox."""
    click.echo(
        "Replace this message by putting your code into " "anomaly_toolbox.cli.main"
    )
    click.echo("See click documentation at http://click.pocoo.org/")

    id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/experiments/" + id

    experiment = ganomaly_experiments.ExperimentMNIST(log_dir)
    experiment.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
