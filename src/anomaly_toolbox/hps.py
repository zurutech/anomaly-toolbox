import itertools
from typing import Dict, List, Tuple

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


def convert_to_hps(raw_hps: Dict[str, Dict]) -> List[hp.HParam]:
    hps = [
        hp.HParam(entry, raw_hps[entry]["type"](raw_hps[entry]["value"]))
        for entry in raw_hps
    ]
    return hps


def grid_search(
    experiment_func,
    experiment_setup,
    log_dir: str,
):
    hps = convert_to_hps(experiment_setup[0])
    metrics = experiment_setup[1]

    # Log the hps
    with tf.summary.create_file_writer(log_dir).as_default():
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
        print(f"--- Starting trial: {run_name}")
        print(hps_run)
        # TODO: actually run the experiment
        experiment_func(
            hps_run,
            log_dir + "/" + run_name,
        )
        session_num += 1
