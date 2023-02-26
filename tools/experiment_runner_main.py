"""Executes a series of experiments described by a config file.

   The config json file format:
   * Contains key-value pairs for any parameters that should take
     non-default values.
   * Uses a list of values to execute a sweep. 
   * Includes "experiment_name_prefix" to construct the
     experiment_name. The empty string is used if this is not
     provided.

   The experiment runner will execute the product of all value lists.
   The experiment_name_prefix will be appended with sweep values
   separated by underscores, if there are any. An empty string is used
   as the experiment_name is none is provided.

   Example config json:
   {
     "experiment-name-prefix" : "sample_sweep",
     "initial-memories-count" : [0, 100, 10000],
     "env-width" : [6, 8],
     "val-check-interval" : 10,
   }

   A total of 6 experimental configurations will be executed. The
   first will use the experiment_name "sample_sweep_6_0", appending
   the first env-width and initial_memories_count values. The ordering
   of the value from "env-width" and the value from
   "initial-memories-count" is determined by sorting the keys.

   Usage:
   $ python -m tools.experiment_runner --config example_config.json

"""

from typing import Any

import argparse
import itertools
import json

_EXPERIMENT_NAME_PREFIX = "experiment_name_prefix"

def _execute_experiment(expected_log: str, test_log: str) -> bool:
    """

    """
    return True

def _extract_sweep_keys_and_values(config: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    keys = []
    value_lists = []
    for key,value in config.items():
        if type(value) == list:
            keys.append(key)
            value_lists.append(value)

    for key in keys:
        del config[key]

def main():
    parser = argparse.ArgumentParser(
        description="Execute a series of experiments."
    )
    parser.add_argument(
        "--config",
        help="The filepath to the json file describing the experimental config.",
        required=True,
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    
    experiment_name_prefix = config[_EXPERIMENT_NAME_PREFIX] if _EXPERIMENT_NAME_PREFIX in config else ''
    del config[_EXPERIMENT_NAME_PREFIX]

    sweep_keys, sweep_values = _extract_sweep_keys_and_values(config)

    print(sweep_values)
    print(sweep_keys)


if __name__ == "__main__":
    main()
