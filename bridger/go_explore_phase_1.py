from gym_bridges.envs.bridges_env import BridgesEnv
import copy
from typing import Any
from bridger import hash_utils
from dataclasses import dataclass
import numpy as np
import multiprocessing
import functools
import torch

from bridger.logging_utils.object_logging import ObjectLogManager
from bridger import config
from bridger.logging_utils.log_entry import SuccessEntry


def _count_score(
    v: float, wa: float, pa: float, epsilon_1: float, epsilon_2: float
) -> int:
    return wa * (1 / (v + epsilon_1)) ** pa + epsilon_2


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
        hparams: Any,
    ):
        self._processes = processes
        self._width = width
        self._env = env
        self._num_iterations = num_iterations
        self._num_actions = num_actions
        self._hparams = hparams
        self.success_entries: set[SuccessEntry] = set()

        _collect_generate_success_entry = functools.partial(
            generate_success_entry,
            self._env,
            self._num_iterations,
            self._num_actions,
            self._hparams,
        )
        seeds: list[int] = list(range(self._processes))

        with multiprocessing.Pool(processes=self._processes) as pool:
            for success_entries in pool.map(_collect_generate_success_entry, seeds):
                self.success_entries.update(set(success_entries))


class StateCache:

    _cache: dict[Any, CacheEntry] = {}
    current_best: int = 10000000

    def __init__(self, rng, hparams):
        self._rng = rng
        self._hparams = hparams

    def update_times_since_led_to_something_new(
        self, state, led_to_something_to_new: bool
    ) -> None:
        key = hash_utils.hash_tensor(state)
        assert key in self._cache
        if led_to_something_to_new:
            self._cache[key].steps_since_led_to_something_new = 0
        else:
            self._cache[key].steps_since_led_to_something_new += 1

    def update_current_best(self, trajectory_length: int):
        #        print("updating with " , trajectory_length, ' ' , self.current_best)
        self.current_best = min(self.current_best, trajectory_length)

    #        print("CURRENT BEST: " , self.current_best)

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

        cache_keys = []
        state_count_scores = []
        for state, cache_entry in self._cache.items():
            cache_keys.append(state)

            steps_since_led_to_something_new_score = _count_score(
                v=cache_entry.steps_since_led_to_something_new,
                wa=hparams.go_explore_wa_led_to_something_new,
                pa=hparams.go_explore_pa,
                epsilon_1=hparams.go_explore_epsilon_1,
                epsilon_2=hparams.go_explore_epsilon_2,
            )
            sampled_score = _count_score(
                v=cache_entry.sampled_count,
                wa=hparams.go_explore_wa_sampled,
                pa=hparams.go_explore_pa,
                epsilon_1=hparams.go_explore_epsilon_1,
                epsilon_2=hparams.go_explore_epsilon_2,
            )
            visited_score = _count_score(
                v=cache_entry.visit_count,
                wa=hparams.go_explore_wa_times_visited,
                pa=hparams.go_explore_pa,
                epsilon_1=hparams.go_explore_epsilon_1,
                epsilon_2=hparams.go_explore_epsilon_2,
            )
            state_count_scores.append(
                steps_since_led_to_something_new_score + sampled_score + visited_score
            )
        state_count_scores_sum = sum(state_count_scores)
        state_count_probs = [x / state_count_scores_sum for x in state_count_scores]

        key_index = self._rng.choice(
            range(len(cache_keys)), size=n, p=state_count_probs
        )[0]
        key = cache_keys[key_index]
        entry = self._cache[key]
        entry.sampled_count += 1
        return (torch.tensor(key[1]).reshape(key[0]), entry)


def generate_success_entry(
    env: BridgesEnv, num_iterations: int, num_actions: int, hparams: Any, seed: int
) -> list[SuccessEntry]:
    rng = np.random.default_rng(seed)
    cache: StateCache = StateCache(rng, hparams)
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
        if len(current_trajectory) >= cache.current_best:
            break

        action = rng.choice(range(env.nA))
        current_trajectory += (action,)
        next_state, reward, done, aux = env.step(action)
        rewards += (reward,)
        if done:
            if aux["is_success"] and all(np.array(rewards) > -0.5):
                #                print('yay' , current_trajectory)
                success_entries.append(
                    SuccessEntry(trajectory=current_trajectory, rewards=rewards)
                )
                led_to_something_new = True
                cache.update_current_best(len(current_trajectory))
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
    num_iterations = hparams.go_explore_num_iterations
    num_actions = hparams.go_explore_num_actions

    success_entry_generator = SuccessEntryGenerator(
        processes=processes,
        width=width,
        env=BridgesEnv(width=width, force_standard_config=True),
        num_iterations=num_iterations,
        num_actions=num_actions,
        hparams=hparams,
    )

    with ObjectLogManager(
        "object_logging", "success_entry", create_experiment_dir=True
    ) as object_logger:
        object_logger.log("success_entry.pkl", success_entry_generator.success_entries)

    print(
        f"==========\nEntry Count: {len(success_entry_generator.success_entries)}\n * wa-sampled: {hparams.go_explore_wa_sampled}\n * wa-new: {hparams.go_explore_wa_led_to_something_new}\n * wa-visit: {hparams.go_explore_wa_times_visited}\nShortest: {sorted([len(x.trajectory) for x in success_entry_generator.success_entries ])}"
    )
