import gym
import itertools
import pickle
import numpy as np
import os

from bridger import hash_utils
from typing import Generator, Any


def n_bricks(
    brick_count: int, env: gym.Env
) -> Generator[tuple[Any, Any, Any, Any, Any], None, None]:
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


class DatasetGenerator:
    def __init__(
        self, log_filename_directory: str, n_bricks: int, k: int, env: gym.Env
    ):
        self._log_filename_directory: str = log_filename_directory
        self._n_bricks: int = n_bricks
        self._k: int = k
        self._bridges: list[np.array] = []
        self._is_bridge: list[bool] = []
        self._is_bridge_and_uses_less_than_k_bricks: list[bool] = []
        self._bridge_height: list[int] = []
        self._brick_counts: list[int] = []
        self._env: gym.Env = env

    def generate_dataset(self):
        for _, _, next_state, _, done in n_bricks(self._n_bricks, self._env):
            brick_count = np.sum(
                [
                    [brick == self._env.StateType.BRICK for brick in row]
                    for row in next_state
                ]
                // 2
            )

            # Calculate the height of the state.
            bridge_height = next_state.shape[0] - 1

            self._bridges.append(next_state)
            self._is_bridge.append(done)
            self._is_bridge_and_uses_less_than_k_bricks.append(
                done and brick_count < self._k
            )
            self._bridge_height.append(bridge_height)
            self._brick_counts.append(brick_count)

            print(
                next_state,
                done,
                done and brick_count < self._k,
                bridge_height,
                brick_count,
            )

    def finalize(self):
        FILENAME_TO_VARIABLE = {
            "bridges": self._bridges,
            "is_bridge": self._is_bridge,
            "is_bridge_and_uses_less_than_k_bricks": self._is_bridge_and_uses_less_than_k_bricks,
            "bridge_height": self._bridge_height,
            "brick_counts": self._brick_counts,
        }

        for filename, lst in FILENAME_TO_VARIABLE.items():
            with open(
                os.path.join(self._log_filename_directory, f"{filename}.pkl"), "wb+"
            ) as f:
                pickle.dump(lst, f)
