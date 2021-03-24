hparam_dict = dict()

key = "tau"
hparam_dict[key] = {"type": float, "default": 0.01}
help_str = "The fraction of the step (from target Q to real Q) taken when updating target Q's parameters."
hparam_dict[key]["help"] = help_str

key = "gamma"
hparam_dict[key] = {"type": float, "default": 0.99}
help_str = "The discount factor used when computing TD targets."
hparam_dict[key]["help"] = help_str

key = "epsilon"
hparam_dict[key] = {"type": float, "default": 0.05}
help_str = (
    "The probability with which the agent explores actions (uniformly at random) at inference time. "
    "This also acts as a lower bound for the epsilon policy, i.e. epsilon cannot decay beyond this during training."
)
hparam_dict[key]["help"] = help_str

key = "epsilon_training_start"
hparam_dict[key] = {"type": float, "default": 1.0}
help_str = "The value of epsilon to use at the start of training."
hparam_dict[key]["help"] = help_str

key = "epsilon_decay_rate"
hparam_dict[key] = {"type": float, "default": 9.5e-4}
help_str = "The rate (per episode) at which epsilon decays"
hparam_dict[key]["help"] = help_str

key = "epsilon_decay_rule"
hparam_dict[key] = {
    "type": str,
    "default": "arithmetic",
    "choices": ["arithmetic", "geometric"],
}
help_str = "The manner in which epsilon decays, e.g. whether it decays arithmetically or geometrically"
hparam_dict[key]["help"] = help_str
