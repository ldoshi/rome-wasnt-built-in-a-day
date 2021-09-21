"""A set of utility functions"""

from typing import Any, Dict
from collections.abc import Callable

from bridger.config import validate_kwargs


def validate_input(module_name: str, config: Dict[str, Dict[str, Any]]) -> Callable:
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
        def wrapper(self, hparams=None, **kwargs):
            if hparams:
                hparams = vars(hparams)
                hparams.update(kwargs)
                kwargs = hparams

            return func(self, validate_kwargs(module_name, config, **kwargs))

        return wrapper

    return decorator_function
