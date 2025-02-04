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
from bridger.logging_utils.log_entry import SuccessEntry
from bridger import config

RNG = 42
NUM_SAMPLES_PER_PROCESS = 10


def _count_score(
    v: float, wa: float, pa: float, epsilon_1: float, epsilon_2: float
) -> int:
    return wa * (1 / (v + epsilon_1)) ** pa + epsilon_2


class SuccessEntryGenerator:
    """
    A generator class for creating success entries using multiple processes.

    This class initializes the parameters required for generating success entries and
    uses these parameters to generate the entries by calling the `generate_success_entry` function.

    Attributes:
        processes (int): The number of parallel processes to use.
        width (int): The width parameter for the environment.
        env (BridgesEnv): The environment in which to perform the rollouts.
        num_iterations (int): The number of iterations to run the exploration.
        num_actions (int): The number of actions to perform in each rollout.
        hparams (Any): Hyperparameters for the exploration process.
    """

    def __init__(
        self,
        processes: int,
        width: int,
        env: BridgesEnv,
        num_iterations: int,
        num_actions: int,
        hparams: Any,
    ):
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
            processes=processes,
        )


@dataclass
class CacheEntry:
    trajectory: tuple[int]
    rewards: tuple[float]
    steps_since_led_to_something_new: int = 0
    sampled_count: int = 0
    visit_count: int = 1


class CellManager:

    def cache_key(self, state: np.ndarray) -> str:
        pass


class StateCellManager(CellManager):

    def cache_key(self, state: np.ndarray) -> str:
        return hash_utils.hash_tensor(state)


def build_cell_manager(hparams) -> CellManager:
    match hparams.cell_manager:
        case "state_cell_manager":
            return StateCellManager()
        case _:
            raise ValueError(
                f"Unrecognized cell manager provided: {hparams.cell_manager}"
            )


class StateCache:

    def __init__(self, rng, hparams, cell_manager: CellManager):
        self.current_best: int = 10000000
        self._cache: dict[Any, CacheEntry] = {}
        self._rng = rng
        self._hparams = hparams
        self._cell_manager = cell_manager

    def update_times_since_led_to_something_new(
        self, state, led_to_something_to_new: bool
    ) -> None:
        key = self._cell_manager.cache_key(state)
        assert key in self._cache
        if led_to_something_to_new:
            self._cache[key].steps_since_led_to_something_new = 0
        else:
            self._cache[key].steps_since_led_to_something_new += 1

    def update_current_best(self, trajectory_length: int):
        self.current_best = min(self.current_best, trajectory_length)

    def visit(
        self, state: np.ndarray, trajectory: tuple[int], rewards: tuple[float]
    ) -> bool:
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
                wa=self._hparams.go_explore_wa_led_to_something_new,
                pa=self._hparams.go_explore_pa,
                epsilon_1=self._hparams.go_explore_epsilon_1,
                epsilon_2=self._hparams.go_explore_epsilon_2,
            )
            sampled_score = _count_score(
                v=cache_entry.sampled_count,
                wa=self._hparams.go_explore_wa_sampled,
                pa=self._hparams.go_explore_pa,
                epsilon_1=self._hparams.go_explore_epsilon_1,
                epsilon_2=self._hparams.go_explore_epsilon_2,
            )
            visited_score = _count_score(
                v=cache_entry.visit_count,
                wa=self._hparams.go_explore_wa_times_visited,
                pa=self._hparams.go_explore_pa,
                epsilon_1=self._hparams.go_explore_epsilon_1,
                epsilon_2=self._hparams.go_explore_epsilon_2,
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
        Update the current cache with values from a new cache.

        This method iterates over the entries in the new cache and updates the corresponding
        entries in the current cache based on the rewards and trajectory lengths. If the new
        cache entry has higher rewards or the same rewards but a shorter trajectory, the current
        cache entry is updated with the new rewards and trajectory. Additionally, the visit count
        and steps since the entry led to something new are accumulated.

        Args:
            new_cache (StateCache): The new cache containing updated state entries.

        Returns:
            None
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
                # TODO (Joseph): Figure out if this is the correct way to update the steps since led to something new.
                cache_entry.steps_since_led_to_something_new += (
                    new_cache_entry.steps_since_led_to_something_new
                )


def rollout(
    env: BridgesEnv,
    num_actions: int,
    cache: StateCache,
    start_state: np.ndarray,
    start_entry: CacheEntry,
    rng: int,
) -> StateCache:
    success_entries: set[SuccessEntry] = set()
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


def generate_success_entry(
    env: BridgesEnv,
    num_iterations: int,
    num_actions: int,
    hparams: Any,
    seed: int,
    processes: int,
) -> set[SuccessEntry]:
    """
    Generate success entries by performing exploration in the given environment.

    This function initializes the state cache and performs exploration using the specified
    number of iterations and actions. It collects successful entries during the exploration
    process and returns them as a set.

    Args:
        env (BridgesEnv): The environment in which to perform the rollouts.
        num_iterations (int): The number of iterations to run the exploration.
        num_actions (int): The number of actions to perform in each rollout.
        hparams (Any): Hyperparameters for the exploration process.
        seed (int): Random seed for initializing the random number generator.

    Returns:
        set[SuccessEntry]: A set of generated success entries.
    """
    rng = np.random.default_rng(seed)
    cell_manager = build_cell_manager(hparams)
    cache: StateCache = StateCache(rng, hparams, cell_manager)
    cache.visit(state=env.reset(), trajectory=tuple(), rewards=tuple())
    success_entries: set[SuccessEntry] = set()
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


def explore(
    rng: np.random.default_rng,
    env: BridgesEnv,
    cache: StateCache,
    num_iterations: int,
    num_actions: int,
    processes: int,
    success_entries: set[SuccessEntry],
) -> None:
    """
    Perform exploration using multiple processes to collect rollouts and update the state cache.

    This function runs a specified number of iterations, where in each iteration, it samples
    start states and entries from the cache, generates random seeds for each process, and
    collects rollouts in parallel using multiprocessing. The collected rollouts are then used
    to update the cache and accumulate successful entries.

    Args:
        rng (int): Random number generator for generating seeds.
        env (BridgesEnv): The environment in which to perform the rollouts.
        cache (StateCache): The state cache to sample from and update.
        num_iterations (int): The number of iterations to run the exploration.
        num_actions (int): The number of actions to perform in each rollout.
        processes (int): The number of parallel processes to use for rollouts.
        success_entries (set[SuccessEntry]): A set to accumulate successful entries.

    Returns:
        None
    """

    for _ in range(num_iterations):
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
                # TODO (Joseph): Figure out how to update the cache with the new cache correctly. Why am I updating the success entries and the cache separately?
                success_entries.update(rollout_success_entries)
                cache.update(rollout_cache)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    parser = config.get_hyperparam_parser(
        config.bridger_config,
        description="Hyperparameter Parser for the BridgeBuilderModel",
        parser=None,
    )
    hparams = parser.parse_args()

    width = hparams.env_width
    num_iterations = hparams.go_explore_num_iterations
    num_actions = hparams.go_explore_num_actions

    success_entry_generator = SuccessEntryGenerator(
        processes=2,
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
