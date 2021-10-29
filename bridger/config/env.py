from argparse import BooleanOptionalAction

hparam_dict = dict()

key = "name"
hparam_dict[key] = {"type": str, "default": "gym_bridges.envs:Bridges-v0"}
help_str = "The name of the Gym environment (with gym_bridges.env) to use."
hparam_dict[key]["help"] = help_str

key = "width"
hparam_dict[key] = {"type": int, "default": 6}
help_str = "The width of the environment to be used."
hparam_dict[key]["help"] = help_str

key = "force_standard_config"
hparam_dict[key] = {"type": bool, "default": True, "action": BooleanOptionalAction}
help_str = "Whether to use only the standard environment configuration."
hparam_dict[key]["help"] = help_str

key = "display"
hparam_dict[key] = {"type": bool, "default": False, "action": BooleanOptionalAction}
help_str = "Whether to render the environment after every action taken"
hparam_dict[key]["help"] = help_str
