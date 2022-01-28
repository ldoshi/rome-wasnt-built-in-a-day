"""A set of utility functions"""

from typing import Any
from collections.abc import Callable

from bridger.config import validate_kwargs


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
