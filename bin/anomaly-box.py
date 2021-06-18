"""Training & evaluation script for anomaly toolbox."""

import sys
from datetime import datetime
from pathlib import Path
import click
import warnings

from anomaly_toolbox.benchmarks import AVAILABLE_BENCHMARKS
from anomaly_toolbox.experiments import AVAILABLE_EXPERIMENTS
from anomaly_toolbox.hps import grid_search


@click.command()
@click.option(
    "chosen_experiment",
    "--experiment",
    help="Experiment to run.",
    type=click.Choice(list(AVAILABLE_EXPERIMENTS.keys()), case_sensitive=False),
)
@click.option(
    "chosen_benchmark",
    "--benchmark",
    help="Benchmark to run.",
    type=click.Choice(list(AVAILABLE_BENCHMARKS.keys()), case_sensitive=False),
)
@click.option(
    "test_run",
    "--test-run",
    help="When running a benchmark, the path of the SavedModel to use.",
    type=str,
)
@click.option(
    "hparams_file_path",
    "--hps-path",
    help="When running a benchmark, the path of the JSON file where all "
    "the hyperparameters are located.",
    type=Path,
)
@click.option(
    "hparams_tuning",
    "--hps-tuning",
    help="If you want to use hyperparameters tuning, use 'True' here. Default is False.",
    type=bool,
    default=False,
)
def main(
    chosen_experiment: str,
    chosen_benchmark: str,
    test_run: str,
    hparams_file_path: Path,
    hparams_tuning: bool,
) -> int:

    # Warning to the user if the hparmas_tuning.json && --hps-tuning==False
    if "tuning" in str(hparams_file_path) and not hparams_tuning:
        warnings.warn(
            "You choose to use the tuning JSON but the tuning boolean ("
            "--hps-tuning) is False. Only one kind of each parameters will be taken "
            "into consideration. No tuning will be performed."
        )

    """Console script for anomaly_toolbox."""
    if chosen_experiment:
        log_dir = (
            Path("logs")
            / "experiments"
            / chosen_experiment
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        experiment = AVAILABLE_EXPERIMENTS[chosen_experiment.lower()](
            hparams_file_path, log_dir
        )
        experiment.run(hparams_tuning, grid_search)
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
