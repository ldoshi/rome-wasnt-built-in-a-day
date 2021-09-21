import argparse
from typing import Any

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
            if "default" in val:
                kwargs[key] = val["default"]
    return kwargs
