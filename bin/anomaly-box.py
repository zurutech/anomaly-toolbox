"""Training & evaluation script for anomaly toolbox."""

import importlib
import logging
import sys
from pathlib import Path
import glob
import os

import json
import click
from tabulate import tabulate

import anomaly_toolbox.datasets as available_datasets
import anomaly_toolbox.experiments as available_experiments
from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.hps import grid_search


@click.command()
@click.option(
    "chosen_experiment",
    "--experiment",
    help="Experiment to run.",
    type=click.Choice(available_experiments.__experiments__, case_sensitive=True),
)
@click.option(
    "hps_path",
    "--hps-path",
    help="When running an experiment, the path of the JSON file where all "
    "the hyperparameters are located.",
    type=Path,
    required=True,
)
@click.option(
    "hps_tuning",
    "--tuning",
    help="If you want to use hyperparameters tuning, use 'True' here. Default is False.",
    type=bool,
    default=False,
)
@click.option(
    "dataset",
    "--dataset",
    help=(
        "The dataset to use. Can be a ready to use dataset, or a .py file "
        "that implements the AnomalyDetectionDataset interface"
    ),
    type=str,
    required=True,
)
@click.option(
    "run_all",
    "--run-all",
    help="Run all the available experiments",
    type=bool,
    default=False,
)
def main(
    chosen_experiment: str,
    hps_path: Path,
    hps_tuning: bool,
    dataset: str,
    run_all: bool,
) -> int:

    # Warning to the user if the hparams_tuning.json && --tuning==False
    if "tuning" in str(hps_path) and not hps_tuning:
        logging.warning(
            "You choose to use the tuning JSON but the tuning boolean ("
            "--tuning) is False. Only one kind of each parameters will be taken "
            "into consideration. No tuning will be performed."
        )

    # Instantiate dataset config from dataset name
    if dataset.endswith(".py"):
        file_path = Path(dataset).absolute()
        name = file_path.stem
        sys.path.append(str(file_path.parent))

        dataset_instance = getattr(__import__(name), name)()
        if not isinstance(dataset_instance, AnomalyDetectionDataset):
            logging.error(
                "Your class %s must implement the "
                "anomaly_toolbox.datasets.dataset.AnomalyDetectionDataset"
                "interface",
                dataset_instance,
            )
            return 1
    else:
        try:
            dataset_instance = getattr(
                importlib.import_module("anomaly_toolbox.datasets"),
                dataset,
            )()
        except (ModuleNotFoundError, AttributeError, TypeError):
            logging.error(
                "Dataset %s is not among the available: %s",
                dataset,
                ",".join(available_datasets.__datasets__),
            )
            return 1

    if run_all and chosen_experiment:
        logging.error("Only one between --run_all and --experiment can be used.")
        return 1

    if not (chosen_experiment or run_all):
        logging.error(
            "Please choose a valid CLI flag.\n%s --help", Path(sys.argv[0]).name
        )
        return 1

    if not hps_path or not hps_path.exists():
        logging.error(
            "Check that %s exists and it's a valid JSON containing the hyperparameters.",
            hps_path,
        )
        return 1

    if chosen_experiment:
        experiments = [chosen_experiment]
    else:
        experiments = available_experiments.__experiments__

    for experiment in experiments:
        log_dir = Path("logs") / experiment
        log_dir.mkdir(parents=True, exist_ok=True)

        try:
            experiment_instance = getattr(
                importlib.import_module("anomaly_toolbox.experiments"),
                experiment,
            )(hps_path, log_dir)
        except (ModuleNotFoundError, AttributeError, TypeError):
            logging.error(
                "Experiment %s is not among the available: %s",
                experiment,
                ",".join(available_experiments.__experiments__),
            )
            return 1

        experiment_instance.run(hps_tuning, grid_search, dataset_instance)

    # Check the best result for all experiments
    experiments = available_experiments.__experiments__
    if run_all:
        result_dirs = ["Experiment", "AUC", "AUPRC", "Precision", "Recall"]

        # For every experiment
        table = [result_dirs]  # This create the table header
        for experiment in experiments:
            table_row = [[experiment, 0.0, 0.0, 0.0, 0.0]]
            log_dir = Path("logs") / experiment / "results" / "best"
            if os.path.exists(log_dir):
                dirs = os.listdir(log_dir)

                # For every possible metric collected
                for dir in dirs:
                    # Check if the dir considered is actually a metric dir
                    if dir in result_dirs:
                        # Get the index inside the list
                        idx = result_dirs.index(dir)
                        json_file = glob.glob(str(log_dir / dir / "result.json"))

                        # Get the value from the json file
                        with open(json_file[0], "r") as file:
                            data = json.load(file)

                        current_result = data["best_on_test_dataset"]

                        # Put the result in the correct position of the list
                        table_row[0][idx] = current_result

                # Put the results in the table
                table = table + table_row

        # TODO: to be deleted, just for output log
        print(tabulate(table, headers="firstrow", tablefmt="github"))

        with open("result_table.md", "w") as outputfile:
            outputfile.write(tabulate(table, headers="firstrow", tablefmt="github"))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
