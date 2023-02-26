"""Iterates a series of experiments described by a config file."""

from typing import Any, Callable, List

import argparse
import copy
import itertools
import json


_EXPERIMENT_NAME_PREFIX = "experiment-name-prefix"
_EXPERIMENT_NAME = "experiment-name"
_FLAG_NAME_FORMAT = """--{flag_name}"""

def _extract_sweep_keys_and_values(config: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    keys = []
    value_lists = []
    for key in sorted(config):
        value = config[key]
        if type(value) == list:
            keys.append(key)
            value_lists.append(value)

    for key in keys:
        del config[key]

def _add_args(args: List[str], flag_name: str, flag_value: str):
    args.append(_FLAG_NAME_FORMAT.format(flag_name=flag_name))
    args.append(flag_value)
        
def run_experiments(config: Any, execute_fn: Callable[[List[str]], None]) -> None:
    config = copy.deepcopy(config)
    args = []
    
    experiment_name_prefix = ''
    if _EXPERIMENT_NAME_PREFIX in config:
        experiment_name_prefix = config[_EXPERIMENT_NAME_PREFIX]
        del config[_EXPERIMENT_NAME_PREFIX]

#    sweep_keys, sweep_values = _extract_sweep_keys_and_values(config)
    experiment_name = experiment_name_prefix

    if experiment_name:
        _add_args(args=args, flag_name=_EXPERIMENT_NAME, flag_value=experiment_name)

    # Fill in remaining args.
    for flag_name in sorted(config):
        flag_value = config[flag_name]
        _add_args(args=args, flag_name=flag_name, flag_value=flag_value)
    
    execute_fn(args)

