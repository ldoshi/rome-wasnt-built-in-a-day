hparam_dict = dict()

key = "save_model_dir"
hparam_dict[key] = {"type": str, "default": "model"}
help_str = "Path to folder in which to save trained model."
hparam_dict[key]["help"] = help_str

key = "checkpoint_model_dir"
hparam_dict[key] = {"type": str, "default": "checkpoints"}
help_str = "Path to folder in which model checkpoints will be saved"
hparam_dict[key]["help"] = help_str

key = "checkpoint_interval"
hparam_dict[key] = {"type": int, "default": 10}
help_str = "Number of training batch steps per checkpoint creation"
hparam_dict[key]["help"] = help_str

key = "interactive_mode"
hparam_dict[key] = {"type": bool, "default": False}
help_str = "Boolean indicating whether to run training interactively"
hparam_dict[key]["help"] = help_str
