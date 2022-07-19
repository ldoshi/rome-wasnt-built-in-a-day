import argparse
from typing import Any
from collections.abc import Callable

from bridger.config.agent import hparam_dict as agent_config
from bridger.config.buffer import hparam_dict as buffer_config
from bridger.config.checkpointing import hparam_dict as checkpoint_config
from bridger.config.env import hparam_dict as env_config
from bridger.config.training import hparam_dict as training_config

# The defaults in this submodule have not yet been debugged or validated (changes
# likely needed).

bridger_config = dict(
    **agent_config,
    **buffer_config,
    **checkpoint_config,
    **{"env_" + k: v for k, v in env_config.items()},
    **training_config,
)


def get_hyperparam_parser(config, description="", parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    for key, kwargs in config.items():
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    return parser


def validate_kwargs(
    module_name: str, config: dict[str, dict[str, Any]], **kwargs
) -> dict[str, Any]:
    """Validates keyword arguments using a config of expected arguments.

    Reports on extraneous inputs, ensures that all required inputs are present,
    and populates with default values for all optional arguments.

    Args:
        module_name: the name of the callable meant to take these inputs
        config: a config dictionary, mapping input names to property dictionaries
             that would be used by argparse.ArgumentParser
    Keyword Args: all the arguments to be validated

    Returns a dictionary containing a validated set of keyword args to pass as
        inputs to the intended callable.
    """
    missing = ",".join([key for key in kwargs if key not in config])
    if missing:
        print(
            "INFO: The following are not recognized hyperparameters "
            f"for (and will be disregarded by) {module_name}: {missing}"
        )
    for key, val in config.items():
        if key not in kwargs:
            check = not val.get("required", False)
            assert check, f"Required argument {key} not provided for {module_name}"
            # The assert below need not be true in principle - if you're hitting
            # this and think your use case case is valid, remove the assert
            has_default = "default" in val
            assert has_default, f"Default not provided for optional argument {key}"
            if has_default:
                kwargs[key] = val["default"]
    return kwargs


def validate_input(module_name: str, config: dict[str, dict[str, Any]]) -> Callable:
    """A function to generate decorators that validate the input for a
       LightningModule constructor.

    By applying the generated decorator to the constructor of a LightningModule,
    the constructor will report on extraneous inputs, ensure that all required
    inputs are present, and populate default values for all optional arguments.

    Args:
        module_name: the name of the callable meant to take these inputs
        config: a config dictionary, mapping input names to property dictionaries
             that would be used by argparse.ArgumentParser

    Returns a decorator that will replace the inputs (an argparse.Namespace or
        keyword arguments) to the constructor with a validated set of inputs,
        as determined by `config`."""

    def decorator_function(func: Callable) -> Callable:
        def wrapper(self, *args, hparams=None, **kwargs):
            if hparams:
                # TODO: Remove disable when pytype catches up with Python3.9
                # pytype: disable=unsupported-operands
                kwargs = vars(hparams) | kwargs
                # pytype: enable=unsupported-operands

            return func(
                self, *args, hparams=validate_kwargs(module_name, config, **kwargs)
            )

        return wrapper

    return decorator_function
