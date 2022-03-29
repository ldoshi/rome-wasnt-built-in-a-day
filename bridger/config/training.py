from argparse import BooleanOptionalAction

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

key = "debug"
hparam_dict[key] = {"type": bool, "default": False, "action": BooleanOptionalAction}
help_str = "Whether to log to TrainingHistory for debugging purposes"
hparam_dict[key]["help"] = help_str

key = "object_logging_dir"
hparam_dict[key] = {"type": str, "default": "object_logging"}
help_str = "Path to folder in which all object logs will be saved"
hparam_dict[key]["help"] = help_str

key = "initial_memories_count"
hparam_dict[key] = {"type": int, "default": hparam_dict["batch_size"]["default"] * 10}
help_str = "The number of memories to initialize the replay buffer with before starting training"
hparam_dict[key]["help"] = help_str

key = "num_workers"
hparam_dict[key] = {"type": int, "default": 1}
help_str = "The number of workers to set for the DataLoader"
hparam_dict[key]["help"] = help_str

key = "gradient_clip_val"
hparam_dict[key] = {"type": float, "default": 0.5}
help_str = "The gradient_clip_val to provide the Lightning Trainer"
hparam_dict[key]["help"] = help_str

key = "val_check_interval"
hparam_dict[key] = {"type": int, "default": 1000}
help_str = "The val_check_interval to provide the Lightning Trainer. Set to an integer value to run validation every n steps (batches) since we have a streaming use-case running a single training epoch."
hparam_dict[key]["help"] = help_str

key = "val_batch_size"
hparam_dict[key] = {"type": int, "default": 1000}
help_str = "The number of episodes to sample in each validation batch"
hparam_dict[key]["help"] = help_str

key = "early_stopping_patience"
hparam_dict[key] = {"type": int, "default": 10}
help_str = (
    "The number of checks with no improvement after which training will be stopped"
)
hparam_dict[key]["help"] = help_str
