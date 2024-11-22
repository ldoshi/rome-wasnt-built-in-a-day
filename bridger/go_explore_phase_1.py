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

RNG = 42
NUM_SAMPLES_PER_PROCESS = 10


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
        seed = RNG
        self.success_entries = generate_success_entry(
            env=env,
            num_iterations=num_iterations,
            num_actions=num_actions,
            hparams=hparams,
            seed=seed,
        )

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
        #    print("updating with " , trajectory_length, ' ' , self.current_best)
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

        key_indices = self._rng.choice(
            range(len(cache_keys)), size=n, p=state_count_probs
        )
        start_states = []
        start_entries = []
        for key_index in key_indices:
            key = cache_keys[key_index]
            entry = self._cache[key]
            entry.sampled_count += 1
            start_states.append(torch.tensor(key[1]).reshape(key[0]))
            start_entries.append(entry)
        return start_states, start_entries

    def update(self, new_cache: "StateCache") -> None:
        """
        Update a cache with values from an updated new cache.
        """
        for new_state, new_cache_entry in new_cache._cache.items():
            for state, cache_entry in self._cache.items():
                if sum(new_cache_entry.rewards) > sum(cache_entry.rewards) or (
                    sum(new_cache_entry.rewards) == sum(cache_entry.rewards)
                    and len(new_cache_entry.trajectory) < len(cache_entry.trajectory)
                ):
                    cache_entry.rewards = new_cache_entry.rewards
                    cache_entry.trajectory = new_cache_entry.trajectory
                cache_entry.visit_count += new_cache_entry.visit_count
                cache_entry.steps_since_led_to_something_new += (
                    new_cache_entry.steps_since_led_to_something_new
                )

def generate_success_entry(
    env: BridgesEnv, num_iterations: int, num_actions: int, hparams: Any, seed: int
) -> list["SuccessEntry"]:
    rng = np.random.default_rng(seed)
    cache: StateCache = StateCache(rng, hparams)
    cache.visit(state=env.reset(), trajectory=tuple(), rewards=tuple())
    success_entries: list[SuccessEntry] = []
    explore(
        rng=np.random.default_rng(RNG),
        env=env,
        cache=cache,
        num_iterations=num_iterations,
        num_actions=num_actions,
        processes=processes,
        success_entries=success_entries,
    )
    return success_entries


def rollout(env, actions, cache, start_state, start_entry, rng) -> StateCache:
    success_entries = set()
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
                success_entries.append(
                    SuccessEntry(trajectory=current_trajectory, rewards=rewards)
                )
                led_to_something_new = True
                cache.update_current_best(len(current_trajectory))
            break

        cache.visit(next_state, current_trajectory, rewards)

    cache.update_times_since_led_to_something_new(start_state, led_to_something_new)
    return success_entries, cache

def explore(rng, env, cache, num_iterations, num_actions, processes, success_entries):
    
    for iteration_number in range(num_iterations):
        # Use unique seed per process and iteration to avoid sampling same states with same seed. 
        seeds = rng.integers(low=0, high=2**31, size=processes)
        start_states, start_entries = cache.sample(seed=RNG, n=RNG*NUM_SAMPLES_PER_PROCESS)
        rngs = map(np.random.default_rng, seeds)
        
        _collect_rollouts = functools.partial(
            rollout,
            env=self._env,
            actions=self._num_actions,
            cache=copy.deepcopy(cache),
        )

def explore(rng, env, cache, num_iterations, num_actions, processes, success_entries):

    for iteration_number in range(num_iterations):
        seeds = rng.integers(low=0, high=2**31, size=processes)
        start_states, start_entries = cache.sample(
            n=processes * NUM_SAMPLES_PER_PROCESS
        )
        rngs = map(np.random.default_rng, seeds)
        _collect_rollouts = functools.partial(
            rollout,
            env,
            num_actions,
            cache,
        )

        success_entries = set()
        with multiprocessing.Pool(processes=processes) as pool:
            for rollout_success_entries, rollout_cache in pool.starmap(
                _collect_rollouts,
                [*zip(start_states, start_entries, rngs)],
            ):
                success_entries.update(rollout_success_entries)
                cache.update(rollout_cache)

if __name__ == "__main__":
    parser = config.get_hyperparam_parser(
        config.bridger_config,
        description="Hyperparameter Parser for the BridgeBuilderModel",
        parser=None,
    )
    hparams = parser.parse_args()

    processes = 2

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

