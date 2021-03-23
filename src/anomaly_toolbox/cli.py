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
    GANomaly().train()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
