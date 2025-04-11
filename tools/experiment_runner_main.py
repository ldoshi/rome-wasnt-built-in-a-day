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
$ python -m tools.experiment_runner_main --config example_config.json

"""

import argparse
import json
import subprocess

from tools import experiment_runner

# The binary to execute with the arguments provided in the config.
_BINARY = "bridge_builder.py"
_BINARY = "go_explore_phase_1.py"


def execute_fn(args: list[str]) -> None:
    command = [_BINARY] + args
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Execute a series of experiments.")
    parser.add_argument(
        "--config",
        help="The filepath to the json file describing the experimental config.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-processes",
        help="The number of experiments to run in parallel.",
        type=int,
        default=12,
    )
    parsed_args = parser.parse_args()

    with open(parsed_args.config) as f:
        config = json.load(f)

    experiment_runner.run_experiments(
        config=config, execute_fn=execute_fn, num_processes=parsed_args.num_processes
    )


if __name__ == "__main__":
    main()
