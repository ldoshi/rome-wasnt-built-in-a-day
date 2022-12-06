# TODO(lyric): Fix imports.

import gym
import itertools
import numpy as npe
import torch

from torch.utils.data import DataLoader
from typing import Any, Callable, Generator, List, Optional, Tuple


from bridger import builder_trainer, config, hash_utils, policies, qfunctions, replay_buffer
from bridger.debug import action_inversion_checker
from bridger.logging import object_logging
from bridger.logging import log_entry

def _only_reset_state(        env: gym.Env) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    for action in range(env.nA):
        state = env.reset()
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward, done

def _standard_configuration_bridge_states(env: gym.Env) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    build_actions = builder_trainer.get_action_inversion_checker_actions_standard_configuration(env.nA)
    build_actions_substrings = [[ build_action[:i] for i in range(len(build_action)+1) ]   for build_action in build_actions ]

    build_orderings = list(itertools.product(*build_actions_substrings))
    # Remove the last member, which represents a completed bridge
    build_orderings =     build_orderings[:-1]
        
    for left_actions, right_actions in build_orderings:
        state = env.reset()
        for action in left_actions + right_actions:
            state, _, _, _ = env.step(action)

        for action in range(env.nA):
            env.reset(state)
            next_state, reward, done, _ = env.step(action)
            yield state, action, next_state, reward, done
                
    
STRATEGY_ONLY_RESET_STATE = "only_reset_state"
STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES = "standard_configuration_bridge_states"
        
_INITIALIZE_REPLAY_BUFFER_BATCH_IDX = -1
                
_STRATEGY_MAP = {
    STRATEGY_ONLY_RESET_STATE : _only_reset_state,
    STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES : _standard_configuration_bridge_states,
}

    

def initialize_replay_buffer(strategy: str,         env: gym.Env, add_new_experience: Callable[[Any, Any, Any, Any, Any], None], state_visit_logger: Optional[object_logging.OccurrenceLogger] = None):

    if strategy not in _STRATEGY_MAP:
        raise ValueError(f"Unrecognized replay buffer initialization strategy: {strategy}. Known values are: default, {', '.join(_STRATEGY_MAP.keys())}")
    for (state, action, next_state, reward, done) in _STRATEGY_MAP[strategy](env=env):
        add_new_experience(state, action, next_state, reward, done)
        if state_visit_logger:
            state_visit_logger.log_occurrence(
                batch_idx=_INITIALIZE_REPLAY_BUFFER_BATCH_IDX, object=torch.from_numpy(state)
            )

