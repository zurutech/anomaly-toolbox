# -*- coding: utf-8 -*-

"""Console script for anomaly_toolbox."""
import logging
import sys
from datetime import datetime

import click
import tensorflow as tf
from anomaly_toolbox.experiments import AVAILABLE_EXPERIMENTS


@click.command()
@click.option(
    "chosen_experiment",
    "--experiment",
    help="Experiment to run",
    type=click.Choice(list(AVAILABLE_EXPERIMENTS.keys()), case_sensitive=False),
)
def main(chosen_experiment: str):
    """Console script for anomaly_toolbox."""
    id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/experiments/" + id

    experiment = AVAILABLE_EXPERIMENTS[chosen_experiment.lower()](log_dir)
    experiment.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
