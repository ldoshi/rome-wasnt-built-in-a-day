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

key = "debug_action_inversion_checker"
hparam_dict[key] = {"type": bool, "default": False, "action": BooleanOptionalAction}
help_str = "Whether to check and log action inversions for debugging purposes"
hparam_dict[key]["help"] = help_str

key = "debug_td_error"
hparam_dict[key] = {"type": bool, "default": False, "action": BooleanOptionalAction}
help_str = "Whether to check and log td errors for debugging purposes"
hparam_dict[key]["help"] = help_str

key = "object_logging_base_dir"
hparam_dict[key] = {"type": str, "default": "object_logging"}
help_str = "Path to the base folder in which all object logs will be saved"
hparam_dict[key]["help"] = help_str

key = "experiment_name"
hparam_dict[key] = {"type": str, "default": ""}
help_str = "The subdirectory name under object_logging_dir for a particular experiment run. The current datetime is prepended to this value to compose the full subdirectory name."
hparam_dict[key]["help"] = help_str

key = "initial_memories_count"
hparam_dict[key] = {"type": int, "default": hparam_dict["batch_size"]["default"] * 10}
help_str = "The number of memories to initialize the replay buffer with before starting training"
hparam_dict[key]["help"] = help_str

key = "initialize_replay_buffer_strategy"
hparam_dict[key] = {"type": str, "default": None}
help_str = "The strategy used to set the contents of the replay buffer without using the policy before starting training. If not provided, the replay buffer is initialized following the epsilon-greedy policy for initial_memories_count steps."
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
hparam_dict[key] = {"type": int, "default": 200}
help_str = "The val_check_interval to provide the Lightning Trainer. Set to an integer value to run validation every n steps (batches) since we have a streaming use-case running a single training epoch."
hparam_dict[key]["help"] = help_str

key = "val_batch_size"
hparam_dict[key] = {"type": int, "default": 10}
help_str = "The number of episodes to sample in each validation batch"
hparam_dict[key]["help"] = help_str

key = "early_stopping_patience"
hparam_dict[key] = {"type": int, "default": 10}
help_str = (
    "The number of checks with no improvement after which training will be stopped"
)
hparam_dict[key]["help"] = help_str

key = "go_explore_success_entries_path"
hparam_dict[key] = {"type": str, "default": "success_entry/success_entry.pkl"}
help_str = "Path to the file containing SuccessEntry members discovered by go_explore."
hparam_dict[key]["help"] = help_str

key = "go_explore_num_actions"
hparam_dict[key] = {"type": int, "default": 8}
help_str = "The number of actions to take in an exploration rollout for go-explore."
hparam_dict[key]["help"] = help_str

key = "go_explore_num_iterations"
hparam_dict[key] = {"type": int, "default": 8}
help_str = "The number of iterations of exploration rollouts for go-explore."
hparam_dict[key]["help"] = help_str

key = "go_explore_epsilon_1"
hparam_dict[key] = {"type": float, "default": 0.001}
help_str = "The epsilon 1 for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "go_explore_epsilon_2"
hparam_dict[key] = {"type": float, "default": 0.00001}
help_str = "The epsilon 2 for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "go_explore_pa"
hparam_dict[key] = {"type": float, "default": 0.5}
help_str = "The exponent for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "go_explore_wa_sampled"
hparam_dict[key] = {"type": float, "default": 0.1}
help_str = "The times a state was sampled weight for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "go_explore_wa_led_to_something_new"
hparam_dict[key] = {"type": float, "default": 0}
help_str = "The times a state was chosen since it led to a new discovery weight for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "go_explore_wa_times_visited"
hparam_dict[key] = {"type": float, "default": 0.3}
help_str = "The times a state was visited weight for go-explore's count score."
hparam_dict[key]["help"] = help_str

key = "jitter"
hparam_dict[key] = {"type": int, "default": 0}
help_str = "The backwards algorithm range in which the state is selected from."
hparam_dict[key]["help"] = help_str

key = "tag"
hparam_dict[key] = {"type": str, "default": ""}
help_str = "A tag to keep notes on the experiment run"
hparam_dict[key]["help"] = help_str
