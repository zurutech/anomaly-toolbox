"""Training & evaluation script for anomaly toolbox."""

import sys
from datetime import datetime
from pathlib import Path

import click

from anomaly_toolbox.benchmarks import AVAILABLE_BENCHMARKS
from anomaly_toolbox.experiments import AVAILABLE_EXPERIMENTS


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
@click.option(
    "test_run",
    "--test-run",
    help="When running a benchmark, the path of the SavedModel to use.",
    type=str,
)
def main(chosen_experiment: str, chosen_benchmark: str, test_run: str) -> int:
    """Console script for anomaly_toolbox."""
    if chosen_experiment:
        log_dir = Path("logs") / "experiments" / chosen_experiment / datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment = AVAILABLE_EXPERIMENTS[chosen_experiment.lower()](log_dir)
        experiment.run()
    elif chosen_benchmark:
        benchmark = AVAILABLE_BENCHMARKS[chosen_benchmark.lower()](run_path=test_run)
        benchmark.load_from_savedmodel().run()
    else:
        exe = sys.argv[0]
        print(f"Please choose a valid CLI flag.\n{exe} --help", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
