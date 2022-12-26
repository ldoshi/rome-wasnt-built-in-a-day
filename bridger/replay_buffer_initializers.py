"""Tools to initialize the replay buffer before beginning training.

When the --initialize_replay_buffer_strategy is set, one common
use-case is to set --inter-training-steps to 0 and test without
generating any further experiences.

"""

import gym
import functools
import itertools
import torch

from typing import Any, Callable, Generator, Optional, Tuple

from bridger import builder_trainer, hash_utils
from bridger.logging_utils import object_logging


def _only_reset_state(
    env: gym.Env,
) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    """Generates every (reset_state, action) pair.

    This is primarily for debugging the full flow from logging to
    Sibyl.

    Args:
      env: A gym for executing population strategies. The current
        implementation assumes that reset() always returns the same
        starting state.

    Returns:
      A generator that yields every experience created by following
      this strategy.

    """

    for action in range(env.nA):
        state = env.reset()
        next_state, reward, done, _ = env.step(action)
        yield state, action, next_state, reward, done


def _standard_configuration_bridge_states(
    env: gym.Env,
) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    """Generates experiences while building the optimal bridge.

    This strategy assumes the current env standard config. Every
    (state, action) for all action permutations that result in the
    optimal bridge are enumerated.

    Args:
      env: A gym for executing population strategies. The current
        implementation assumes that reset() always returns the same
        starting state.

    Returns:
      A generator that yields every experience created by following
      this strategy.

    """

    build_actions = (
        builder_trainer.get_action_inversion_checker_actions_standard_configuration(
            env.nA
        )
    )
    build_actions_substrings = [
        [build_action[:i] for i in range(len(build_action) + 1)]
        for build_action in build_actions
    ]

    build_orderings = list(itertools.product(*build_actions_substrings))
    # Remove the last member, which represents a completed bridge.
    build_orderings = build_orderings[:-1]

    for left_actions, right_actions in build_orderings:
        state = env.reset()
        for action in left_actions + right_actions:
            state, _, _, _ = env.step(action)

        for action in range(env.nA):
            env.reset(state)
            next_state, reward, done, _ = env.step(action)
            yield state, action, next_state, reward, done


def _n_bricks(
    brick_count: int, env: gym.Env
) -> Generator[Tuple[Any, Any, Any, Any, Any], None, None]:
    """Generates distinct experiences from exhaustively placing n bricks.

    Enumerates every permutation of placing n bricks and removes
    duplicate (state, action) pairs.

    Args:
      brick_count: The number of bricks to place.
      env: A gym for executing population strategies. The current
        implementation assumes that reset() always returns the same
        starting state.

    Returns:
      A generator that yields every experience created by following
      this strategy.
    """

    state_action_tuples = set()
    for episode_actions in itertools.product(range(env.nA), repeat=brick_count):
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
    STRATEGY_ONLY_RESET_STATE: _only_reset_state,
    STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES: _standard_configuration_bridge_states,
    STRATEGY_2_BRICKS: functools.partial(_n_bricks, brick_count=2),
    STRATEGY_4_BRICKS: functools.partial(_n_bricks, brick_count=4),
    STRATEGY_6_BRICKS: functools.partial(_n_bricks, brick_count=6),
}


def initialize_replay_buffer(
    strategy: str,
    replay_buffer_capacity: int,
    env: gym.Env,
    add_new_experience: Callable[[Any, Any, Any, Any, Any, Any], None],
    state_visit_logger: Optional[object_logging.OccurrenceLogger] = None,
    state_logger: Optional[object_logging.LoggerAndNormalizer] = None,
) -> None:
    """Initializes the replay buffer following the provided strategy.

    Args:
      strategy: The name of the strategy to follow.
      replay_buffer_capacity: The capacity of the replay buffer being
        populated.
      env: A gym for executing population strategies. The current
        implementation assumes that reset() always returns the same
        starting state.
      add_new_experience: The function to call in order to add an
        additional experience to the replay buffer.
      state_visit_logger: If provided, any states traversed by the
        population policy are logged as visited.
      state_logger: If provided, provides access to state ids. This
        must be provided to access state_ids, which in turn must be
        passed to add_new_experience if the replay buffer is
        initialzed in debug mode.

    Raises:
      ValueError: If the strategy generates more experiences than the
        capacity of the replay buffer or if the strategy does not
        correspond to a known implementation.

    """

    if strategy not in _STRATEGY_MAP:
        raise ValueError(
            f"Unrecognized replay buffer initialization strategy: {strategy}. Known values are: default, {', '.join(_STRATEGY_MAP.keys())}"
        )

    experience_count = 0
    for (state, action, next_state, reward, done) in _STRATEGY_MAP[strategy](env=env):

        if state_visit_logger:
            state_visit_logger.log_occurrence(
                batch_idx=_INITIALIZE_REPLAY_BUFFER_BATCH_IDX,
                object=torch.from_numpy(state),
            )

        state_id = (
            state_logger.get_logged_object_id(torch.from_numpy(state))
            if state_logger
            else None
        )
        add_new_experience(state, action, next_state, reward, done, state_id)
        experience_count += 1

        if experience_count > replay_buffer_capacity:
            raise ValueError(
                f"Replay buffer initialized with more experiences than capacity ({replay_buffer_capacity})"
            )
