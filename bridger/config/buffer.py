hparam_dict = dict()

key = "capacity"
hparam_dict[key] = {"type": int, "default": 10000}
help_str = "Max number of samples to hold in replay buffer."
hparam_dict[key]["help"] = help_str

key = "alpha"
hparam_dict[key] = {"type": float, "default": 0.5}
help_str = "The exponent to use when transforming absolute TD error into sampling probabilities for the replay buffer."
hparam_dict[key]["help"] = help_str

key = "beta_training_start"
hparam_dict[key] = {"type": float, "default": 0.5}
help_str = (
    "The value of beta (importance sampling exponent) to use at the start of training."
)
hparam_dict[key]["help"] = help_str

key = "beta_training_end"
hparam_dict[key] = {"type": float, "default": 1.0}
help_str = "An upper bound for the value of beta (importance sampling exponent) during training."
hparam_dict[key]["help"] = help_str

key = "beta_growth_rate"
hparam_dict[key] = {"type": float, "default": 5e-4}
help_str = "The rate (per training batch) at which beta changes"
hparam_dict[key]["help"] = help_str

key = "beta_growth_rule"
hparam_dict[key] = {
    "type": str,
    "default": "arithmetic",
    "choices": ["arithmetic", "geometric"],
}
help_str = "The manner in which beta decays, e.g. whether it decays arithmetically or geometrically"
hparam_dict[key]["help"] = help_str
