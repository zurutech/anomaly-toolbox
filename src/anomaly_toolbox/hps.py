"""Common utilities for HPS search."""

import itertools
from typing import List
from pathlib import Path
import json
import sys

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


def hparam_parser(
    hparams_path: Path, experiment: str, hyperparameters_names: List[str]
) -> List[hp.HParam]:
    """
    JSON hyperparameters file parser.

    Args:
        hparams_path: The path of the JSON hparams file.
        experiment: The name of the experiment.
        hyperparameters_names: The list of hyperparameters to be used.

    Returns:
        A list of hyperparameters.
    """
    with open(hparams_path) as f:
        data = json.load(f)

    # Get the experiment name
    experiment_data = data[experiment]

    # FIll the hparam list
    hps: List[hp.HParam] = []
    for hparam_name in hyperparameters_names:

        # Current parameter object
        try:
            current_param = experiment_data[hparam_name]
        except KeyError:
            print(
                f"ERROR: '{hparam_name}' key does not exist inside the {hparams_path} file, "
                f"please check."
            )
            sys.exit(1)

        # Get the 'hp' attribute taking the correct "type" from the JSON object (usually the
        # type is 'Discrete')
        hp_attr = getattr(hp, current_param["type"])

        # Fill the list with the hparam name and the value of the correct type (e.g., 'Discrete')
        hps.append(
            hp.HParam(
                hparam_name,
                hp_attr(current_param["values"]),
            )
        )

    return hps


def grid_search(
    experiment_func,
    hps: List[hp.HParam],
    metrics: List[hp.Metric],
    log_dir: str,
) -> None:
    """
    Perform a grid search over the values of the hyperparameters passed.

    Given a set of hyperparameters, this function firstly creates their combinatorial product
    and then iterates over the result each time calling the given experiment_func.

    Args:
        experiment_func: Callable, usually coding a single run of the experiment
        hps: List of hyperparameters, encoded as hp.HParam
        metrics: List of metrics to log, encoded as hp.Metric
        log_dir: Log directory where the tf.summary.SummaryWriter will save data
    """
    # Log the hps
    with tf.summary.create_file_writer(str(log_dir)).as_default():
        hp.hparams_config(hparams=hps, metrics=metrics)

    # Build all combinations of the HPS values
    hps_values = []
    for entry in hps:
        if type(entry == hp.Discrete):
            hps_values.append(entry.domain.values)
        # TODO: Add support for other type of hps
    hps_combinations = itertools.product(*hps_values)

    # Then for each set of HPS run an experiment
    session_num = 0
    for hps_values in hps_combinations:
        hps_run = {}
        for i, entry in enumerate(hps):
            hps_run[entry.name] = hps_values[i]
        run_name = f"run-{session_num}"
        print(f"--- Starting trial: {run_name} ---")
        print(hps_run)
        experiment_func(
            hps_run,
            str(log_dir) + "/" + run_name,
        )
        session_num += 1
