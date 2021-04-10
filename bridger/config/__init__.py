from config.agent import hparam_dict as agent_config
from config.buffer import hparam_dict as buffer_config
from config.checkpointing import hparam_dict as checkpoint_config
from config.env import hparam_dict as env_config
from config.training import hparam_dict as training_config

bridger_config = dict(
    **agent_config,
    **buffer_config,
    **checkpoint_config,
    **{"env_" + k: v for k, v in env_config.items()},
    **training_config
)


def get_hyperparam_parser(config, description="", parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description=description)
    for key, kwargs in config.items():
        parser.add_argument("--" + key.replace("_", "-"), **kwargs)
    return parser
