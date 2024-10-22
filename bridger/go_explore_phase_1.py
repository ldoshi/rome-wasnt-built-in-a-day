from gym_bridges.envs.bridges_env import BridgesEnv
import copy
from typing import Any
from bridger import hash_utils
from dataclasses import dataclass
import numpy as np
import multiprocessing
import functools
import torch
import json

from bridger.logging_utils.object_logging import ObjectLogManager
from bridger import config
from bridger.logging_utils.log_entry import SuccessEntry


@dataclass
class CacheEntry:
    trajectory: tuple[int]
    rewards: tuple[float]
    steps_since_led_to_something_new: int = 0
    sampled_count: int = 0
    visit_count: int = 1


class SuccessEntryGenerator:
    def __init__(
        self,
        processes: int,
        width: int,
        env: BridgesEnv,
        num_iterations: int,
        num_actions: int,
    ):
        self._processes = processes
        self._width = width
        self._env = env
        self._num_iterations = num_iterations
        self._num_actions = num_actions
        self.success_entries: set[SuccessEntry] = set()

        _collect_generate_success_entry = functools.partial(
            generate_success_entry,
            self._env,
            self._num_iterations,
            self._num_actions,
        )
        seeds: list[int] = list(range(self._processes))

        with multiprocessing.Pool(processes=self._processes) as pool:
            for success_entries in pool.map(_collect_generate_success_entry, seeds):
                self.success_entries.update(set(success_entries))


class StateCache:

    _cache: dict[Any, CacheEntry] = {}

    def __init__(self, rng):
        self._rng = rng

    def update_times_since_led_to_something_new(
        self, state, led_to_something_to_new: bool
    ) -> None:
        key = hash_utils.hash_tensor(state)
        assert key in self._cache
        if led_to_something_to_new:
            self._cache[key].steps_since_led_to_something_new = 0
        else:
            self._cache[key].steps_since_led_to_something_new += 1

    def visit(self, state, trajectory: tuple[int], rewards: tuple[float]) -> bool:
        key = hash_utils.hash_tensor(state)
        if key in self._cache:
            entry = self._cache[key]
            entry.visit_count += 1
            if sum(rewards) > sum(entry.rewards) or (
                sum(rewards) == sum(entry.rewards)
                and len(trajectory) < len(entry.trajectory)
            ):
                entry.rewards = rewards
                entry.trajectory = trajectory
        else:
            self._cache[key] = CacheEntry(trajectory=trajectory, rewards=rewards)

    def sample(self, n=1):
        # Fix this when n > 1.
        assert n == 1
        cache_keys = list(self._cache)
        key_index = self._rng.choice(range(len(cache_keys)), n)[0]
        key = cache_keys[key_index]
        entry = self._cache[key]
        entry.sampled_count += 1
        return (torch.tensor(key[1]).reshape(key[0]), entry)


def generate_success_entry(
    env: BridgesEnv, num_iterations: int, num_actions: int, seed: int
) -> list[SuccessEntry]:
    rng = np.random.default_rng(seed)
    cache: StateCache = StateCache(rng)
    cache.visit(state=env.reset(), trajectory=tuple(), rewards=tuple())
    success_entries: list[SuccessEntry] = []

    explore(rng, env, cache, num_iterations, num_actions, success_entries)

    return success_entries


def rollout(rng, env, start_state, start_entry, cache, num_actions, success_entries):
    env.reset(start_state)

    current_trajectory = copy.deepcopy(start_entry.trajectory)
    rewards: tuple[float] = start_entry.rewards

    led_to_something_new = False
    for _ in range(num_actions):
        action = rng.choice(range(env.nA))
        current_trajectory += (action,)
        next_state, reward, done, _ = env.step(action)
        rewards += (reward,)
        if done:
            success_entries.append(
                SuccessEntry(trajectory=current_trajectory, rewards=rewards)
            )
            led_to_something_new = True
            break

        cache.visit(next_state, current_trajectory, rewards)

    cache.update_times_since_led_to_something_new(start_state, led_to_something_new)


def explore(rng, env, cache, num_iterations, num_actions, success_entries):

    for _ in range(num_iterations):
        start_state, start_entry = cache.sample()
        rollout(rng, env, start_state, start_entry, cache, num_actions, success_entries)


if __name__ == "__main__":
    parser = config.get_hyperparam_parser(
        config.bridger_config,
        description="Hyperparameter Parser for the BridgeBuilderModel",
        parser=None,
    )
    hparams = parser.parse_args()

    processes = 1
    width = hparams.env_width
    num_iterations = hparams.max_training_batches
    num_actions = hparams.go_explore_num_actions

    success_entry_generator = SuccessEntryGenerator(
        processes=processes,
        width=width,
        env=BridgesEnv(width=width, force_standard_config=True),
        num_iterations=num_iterations,
        num_actions=num_actions,
    )

    with open("object_logging/success_entry.json", 'w') as f:
        json.dump(success_entry_generator.success_entries, f, indent=4)

