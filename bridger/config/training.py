hparam_dict = dict()

key = "inter_training_steps"
hparam_dict[key] = {"type": int, "default": 1}
help_str = "The number of steps (state -> new state) to be run between consecutive training batches."
hparam_dict[key]["help"] = help_str

key = "max_training_batches"
hparam_dict[key] = {"type": int, "default": 1000}
help_str = "The max number of training batches to run."
hparam_dict[key]["help"] = help_str

key = "max_episode_length"
hparam_dict[key] = {"type": int, "default": 10}
help_str = "The max number of steps to be run in a single episode."
hparam_dict[key]["help"] = help_str

key = "batch_size"
hparam_dict[key] = {"type": int, "default": 100}
help_str = "The number of transitions to sample in each training batch"
hparam_dict[key]["help"] = help_str

key = "lr"
hparam_dict[key] = {"type": float, "default": 0.02}
help_str = "Learning rate, default is 0.02"
hparam_dict[key]["help"] = help_str

key = "update_bound"
hparam_dict[key] = {"type": float, "default": 1.0}
help_str = "The max absolute value to which to clip each temporal difference (TD) error during training"
hparam_dict[key]["help"] = help_str

key = "seed"
hparam_dict[key] = {"type": int, "default": 42}
help_str = "Random seed for training"
hparam_dict[key]["help"] = help_str
