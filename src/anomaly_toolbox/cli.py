# -*- coding: utf-8 -*-

"""Console script for anomaly_toolbox."""

import sys
from datetime import datetime

import click

from anomaly_toolbox.experiments import AVAILABLE_EXPERIMENTS
from anomaly_toolbox.benchmarks import AVAILABLE_BENCHMARKS

# TODO: move to cli arg
TEST_RUN = (
    "/home/ubik/Documents/work/anomaly-toolbox/logs/experiments/20210603-040934/run-0"
)


@click.command()
@click.option(
    "chosen_experiment",
    "--experiment",
    help="Experiment to run",
    type=click.Choice(list(AVAILABLE_EXPERIMENTS.keys()), case_sensitive=False),
)
@click.option(
    "chosen_benchmark",
    "--benchmark",
    help="Benchmark to run",
    type=click.Choice(list(AVAILABLE_BENCHMARKS.keys()), case_sensitive=False),
)
def main(chosen_experiment: str, chosen_benchmark: str):
    """Console script for anomaly_toolbox."""
    id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/experiments/" + id

    if chosen_experiment:
        experiment = AVAILABLE_EXPERIMENTS[chosen_experiment.lower()](log_dir)
        experiment.run()
    if chosen_benchmark:
        benchmark = AVAILABLE_BENCHMARKS[chosen_benchmark.lower()](run_path=TEST_RUN)
        benchmark.load_from_savedmodel().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
