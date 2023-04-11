"""Iterates a series of experiments described by a config file."""

from typing import Any, Callable, List, Sequence

import argparse
import copy
import itertools
import json


_EXPERIMENT_NAME_PREFIX = "experiment-name-prefix"
_EXPERIMENT_NAME = "experiment-name"
_FLAG_NAME_FORMAT = """--{flag_name}"""
_FALSE_BOOLEAN_FLAG_NAME_FORMAT = """--no-{flag_name}"""


def _extract_sweep_keys_and_values(
    config: dict[str, Any]
) -> tuple[list[str], list[list[Any]]]:
    keys = []
    value_lists = []
    for key in sorted(config):
        value = config[key]
        if type(value) == list:
            keys.append(key)
            value_lists.append(value)

    for key in keys:
        del config[key]

    return keys, value_lists


def _add_args(
    args: list[str], flag_name: str, flag_value: Union[bool, int, float, str]
) -> None:
    """Appends the flag name and value to the args list.

    Boolean flags require special treatment. Only the flag name is
    provided for true and no-{flag name} is provided for false.

    Args:
      args: The args list to which to add the flag. It will be modified.
      flag_name: The name of the flag to add.
      flag_value: The value of the flag to add.

    """
    if type(flag_value) == bool:
        if flag_value:
            args.append(_FLAG_NAME_FORMAT.format(flag_name=flag_name))
        else:
            args.append(_FALSE_BOOLEAN_FLAG_NAME_FORMAT.format(flag_name=flag_name))
        return

    args.append(_FLAG_NAME_FORMAT.format(flag_name=flag_name))
    args.append(str(flag_value))


def _get_experiment_name(experiment_name_prefix: str, values: Sequence) -> str:
    experiment_name_suffix = "_".join([str(x) for x in values])
    return (
        f"{experiment_name_prefix}_{experiment_name_suffix}"
        if experiment_name_prefix
        else experiment_name_suffix
    )


def run_experiments(
    config: dict[str, Any], execute_fn: Callable[[List[str]], None]
) -> None:
    """Calls execute_fn for argument combinations described in config.

    Args:
      config: The config describing parameter values to set as arguments to
        execute_fn. If a value is a list, it describes a series of values
        to iterate over. This function executes cross-product of all
        lists.
      execute_fn: The function to call with each complete set of
        arguments described in the config. The execute_fn should accept
        a list of args where the list contents alternates between a
        parameter flag name (with a leading --) and the corresponding
        value for that flag. All values are cast to str. For example, it
        may be called with the following list:
        ['--env-width', 5, '--initial-memory-count', '100'].

    """

    config = copy.deepcopy(config)
    args = []

    experiment_name_prefix = ""
    if _EXPERIMENT_NAME_PREFIX in config:
        experiment_name_prefix = config[_EXPERIMENT_NAME_PREFIX]
        del config[_EXPERIMENT_NAME_PREFIX]

    sweep_keys, sweep_values = _extract_sweep_keys_and_values(config)

    # Fill in remaining args.
    for flag_name in sorted(config):
        flag_value = config[flag_name]
        _add_args(args=args, flag_name=flag_name, flag_value=flag_value)

    if sweep_keys:
        # Iterate sweep combinations.
        for iteration_values in itertools.product(*sweep_values):
            iteration_args = copy.deepcopy(args)
            assert len(sweep_keys) == len(iteration_values)
            for flag_name, flag_value in zip(sweep_keys, iteration_values):
                _add_args(
                    args=iteration_args, flag_name=flag_name, flag_value=flag_value
                )

            _add_args(
                args=iteration_args,
                flag_name=_EXPERIMENT_NAME,
                flag_value=_get_experiment_name(
                    experiment_name_prefix=experiment_name_prefix,
                    values=iteration_values,
                ),
            )

            execute_fn(iteration_args)
    else:
        if experiment_name_prefix:
            _add_args(
                args=args, flag_name=_EXPERIMENT_NAME, flag_value=experiment_name_prefix
            )
        execute_fn(args)
