# TODO(lyric): Fix imports.

import gym
import functools
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
    # Remove the last member, which represents a completed bridge. 
    build_orderings =     build_orderings[:-1]
        
    for left_actions, right_actions in build_orderings:
        state = env.reset()
        for action in left_actions + right_actions:
            state, _, _, _ = env.step(action)

        for action in range(env.nA):
            env.reset(state)
            next_state, reward, done, _ = env.step(action)
            yield state, action, next_state, reward, done

            
def _n_bricks(brick_count: int, env: gym.Env) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    state_action_tuples = set()
    for episode_actions in itertools.product(range(env.nA) , repeat=brick_count):
        state = env.reset()
        for action in episode_actions:
            next_state, reward, done, _ = env.step(action)

            state_action_tuple = (hash_utils.hash_tensor(state), action)
            if state_action_tuple not in state_action_tuples:
                state_action_tuples.add(state_action_tuple)
                yield state, action, next_state, reward, done

            if done:
                break
               
            state = next_state
               

            
STRATEGY_ONLY_RESET_STATE = "only_reset_state"
STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES = "standard_configuration_bridge_states"
STRATEGY_2_BRICKS = "2_bricks"
STRATEGY_4_BRICKS = "4_bricks"
STRATEGY_6_BRICKS = "6_bricks"
        
_INITIALIZE_REPLAY_BUFFER_BATCH_IDX = -1
                
_STRATEGY_MAP = {
    STRATEGY_ONLY_RESET_STATE : _only_reset_state,
    STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES : _standard_configuration_bridge_states,
    STRATEGY_2_BRICKS : functools.partial(_n_bricks, brick_count=2),
    STRATEGY_4_BRICKS : functools.partial(_n_bricks, brick_count=4),
    STRATEGY_6_BRICKS : functools.partial(_n_bricks, brick_count=6)
}
    

def initialize_replay_buffer(strategy: str, replay_buffer_capacity: int,         env: gym.Env, add_new_experience: Callable[[Any, Any, Any, Any, Any], None], state_visit_logger: Optional[object_logging.OccurrenceLogger] = None):

    if strategy not in _STRATEGY_MAP:
        raise ValueError(f"Unrecognized replay buffer initialization strategy: {strategy}. Known values are: default, {', '.join(_STRATEGY_MAP.keys())}")

    experience_count = 0
    for (state, action, next_state, reward, done) in _STRATEGY_MAP[strategy](env=env):
        add_new_experience(state, action, next_state, reward, done)
        experience_count += 1
        
        if state_visit_logger:
            state_visit_logger.log_occurrence(
                batch_idx=_INITIALIZE_REPLAY_BUFFER_BATCH_IDX, object=torch.from_numpy(state)
            )

        if experience_count > replay_buffer_capacity:
            raise ValueError(f"Replay buffer initialized with more experiences than capacity ({replay_buffer_capacity})")

