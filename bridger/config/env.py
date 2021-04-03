hparam_dict = dict()

key = "name"
hparam_dict[key] = {"type": str, "default": 1}
help_str = "The name of the Gym environment (with gym_bridges.env) to use."
hparam_dict[key]["help"] = help_str

key = "height"
hparam_dict[key] = {"type": int, "default": 4}
help_str = "The height of the environment to be used."
hparam_dict[key]["help"] = help_str

key = "width"
hparam_dict[key] = {"type": int, "default": 6}
help_str = "The width of the environment to be used."
hparam_dict[key]["help"] = help_str

key = "vary_heights"
hparam_dict[key] = {"type": bool, "default": True}
help_str = "Whether the left and right heights of the bridge can be different."
hparam_dict[key]["help"] = help_str

key = "display"
hparam_dict[key] = {"type": bool, "default": False}
help_str = "Whether to render the environment after every action taken"
hparam_dict[key]["help"] = help_str
