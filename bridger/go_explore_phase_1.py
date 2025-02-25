from gym_bridges.envs.bridges_env import BridgesEnv
import copy
import pickle
from typing import Any
from bridger import hash_utils
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
import multiprocessing
import functools
import torch
import io
import gzip

from bridger.logging_utils.object_logging import ObjectLogManager
from bridger.logging_utils.log_entry import SuccessEntry, OccurrenceLogEntry
from bridger import config

# A constant instead of a config because this merely needs to be
# sufficient for decent performance.
NUM_SAMPLES_PER_PROCESS = 100

RolloutParams = namedtuple("RolloutParams", ["env_width", "num_actions"])


def _count_score(
    v: float, wa: float, pa: float, epsilon_1: float, epsilon_2: float
) -> int:
    return wa * (1 / (v + epsilon_1)) ** pa + epsilon_2


@dataclass
class CacheEntry:
    trajectory: tuple[int]
    rewards: tuple[float]
    state_representative_encoded: bytes  # Store compressed bytes
    steps_since_led_to_something_new: int = 0
    sampled_count: int = 0
    visit_count: int = 1

    @property
    def state_representative(self) -> torch.Tensor:
        """Decompress and reconstruct the integer tensor."""
        buffer = io.BytesIO(gzip.decompress(self.state_representative_encoded))
        return torch.tensor(np.load(buffer, allow_pickle=False))

    @staticmethod
    def encode_state_representative(tensor: torch.Tensor) -> bytes:
        """Compress and encode an integer tensor efficiently."""
        buffer = io.BytesIO()
        np.save(buffer, tensor.numpy(), allow_pickle=False)  # Efficient integer storage
        return gzip.compress(buffer.getvalue())  # Further compression


class CellManager:

    def cache_key(self, state: np.ndarray) -> str:
        pass


class StateCellManager(CellManager):

    def cache_key(self, state: np.ndarray) -> str:
        return hash(hash_utils.hash_tensor(state))


class DownsampleCellManager(CellManager):

    def __init__(self, factor_x: int, factor_y: int):
        self.factor_x = factor_x
        self.factor_y = factor_y

    def _downsample_2d(self, state: np.ndarray):
        if state.shape[0] % self.factor_x != 0 or state.shape[1] % self.factor_y != 0:
            raise ValueError(
                "Array dimensions must be divisible by the downsampling factors"
            )

        return state.reshape(
            state.shape[0] // self.factor_x,
            self.factor_x,
            state.shape[1] // self.factor_y,
            self.factor_y,
        ).sum(axis=(1, 3))

    def cache_key(self, state: np.ndarray) -> str:
        return hash(hash_utils.hash_tensor(self._downsample_2d(state)))


# python go_explore_phase_1.py --env-width=4 --go-explore-num-iterations=8 --cell-manager=downsample_cell_manager


def build_cell_manager(hparams) -> CellManager:
    match hparams.cell_manager:
        case "state_cell_manager":
            return StateCellManager()
        case "downsample_cell_manager":
            # TODO(lyric): Add the factors to the config.
            return DownsampleCellManager(2, 2)
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
        key = self._cell_manager.cache_key(state)
        if key in self._cache:
            entry = self._cache[key]
            entry.visit_count += 1
            if sum(rewards) > sum(entry.rewards) or (
                sum(rewards) == sum(entry.rewards)
                and len(trajectory) < len(entry.trajectory)
            ):
                entry.rewards = rewards
                entry.trajectory = trajectory
                entry.state_representative_encoded = (
                    CacheEntry.encode_state_representative(state)
                )
        else:
            self._cache[key] = CacheEntry(
                trajectory=trajectory,
                rewards=rewards,
                state_representative_encoded=CacheEntry.encode_state_representative(
                    state
                ),
            )

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
        start_entries = []
        for key_index in key_indices:
            entry = self._cache[cache_keys[key_index]]
            entry.sampled_count += 1
            start_entries.append(entry)
        return start_entries

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
        for new_cache_key, new_cache_entry in new_cache._cache.items():
            if new_cache_key in self._cache:
                cache_entry = self._cache[new_cache_key]
                if sum(new_cache_entry.rewards) > sum(cache_entry.rewards) or (
                    sum(new_cache_entry.rewards) == sum(cache_entry.rewards)
                    and len(new_cache_entry.trajectory) < len(cache_entry.trajectory)
                ):
                    cache_entry.rewards = new_cache_entry.rewards
                    cache_entry.trajectory = new_cache_entry.trajectory
                    cache_entry.state_representative_encoded = (
                        new_cache_entry.state_representative_encoded
                    )
                cache_entry.visit_count += new_cache_entry.visit_count
                # TODO (Joseph): Figure out if this is the correct way to update the steps since led to something new.
                cache_entry.steps_since_led_to_something_new += (
                    new_cache_entry.steps_since_led_to_something_new
                )

            else:
                # Add to the cache if the state is not already in the cache.
                self._cache[new_cache_key] = new_cache_entry


def rollout(
    rollout_params: RolloutParams,
    cache: StateCache,
    start_entries: list[CacheEntry],
    rngs: list[int],
) -> StateCache:
    success_entries: set[SuccessEntry] = set()

    env = BridgesEnv(width=rollout_params.env_width, force_standard_config=True)

    for start_entry, rng in zip(start_entries, rngs):
        env.reset(start_entry.state_representative)
        current_trajectory = copy.deepcopy(start_entry.trajectory)
        rewards: tuple[float] = start_entry.rewards

        led_to_something_new = False
        for _ in range(rollout_params.num_actions):
            if len(current_trajectory) >= cache.current_best:
                break

            action = rng.choice(range(env.nA))
            current_trajectory += (action,)
            next_state, reward, done, aux = env.step(action)
            rewards += (reward,)
            if done:
                if aux["is_success"] and all(np.array(rewards) > -0.5):
                    success_entries.add(
                        SuccessEntry(trajectory=current_trajectory, rewards=rewards)
                    )
                    led_to_something_new = True
                    cache.update_current_best(len(current_trajectory))
                break

            cache.visit(next_state, current_trajectory, rewards)

        cache.update_times_since_led_to_something_new(
            start_entry.state_representative, led_to_something_new
        )

    return success_entries, cache


def _chunk_list(elements: list[Any], count: int) -> list[list[Any]]:
    if count <= 0:
        raise ValueError("Count must be greater than 0")

    n = len(elements)
    sublist_size, remainder = divmod(n, count)

    result = []
    start = 0
    for i in range(count):
        extra = 1 if i < remainder else 0  # Distribute remainder elements
        end = start + sublist_size + extra
        result.append(elements[start:end])
        start = end

    return result


def explore(
    object_logger: ObjectLogManager,
    hparams: Any,
) -> set[SuccessEntry]:
    """
    Generate success entries by performing exploration in the environment.

    Uses multiple processes to collect rollouts and update the state cache.

    This function runs a specified number of iterations, where in each iteration, it samples
    start states and entries from the cache, generates random seeds for each process, and
    collects rollouts in parallel using multiprocessing. The collected rollouts are then used
    to update the cache and accumulate successful entries.

    Returns:
        set[SuccessEntry]: A set of generated success entries.
    """

    rng = np.random.default_rng(hparams.seed)

    cell_manager = build_cell_manager(hparams)
    cache: StateCache = StateCache(rng, hparams, cell_manager)
    env = BridgesEnv(width=hparams.env_width, force_standard_config=True)
    cache.visit(state=env.reset(), trajectory=tuple(), rewards=tuple())

    rollout_params = RolloutParams(
        env_width=hparams.env_width, num_actions=hparams.go_explore_num_actions
    )

    success_entries: set[SuccessEntry] = set()
    for iteration in range(hparams.go_explore_num_iterations):
        start_entries = cache.sample(
            n=hparams.go_explore_num_processes * NUM_SAMPLES_PER_PROCESS
        )
        object_logger.log(
            "start_entries.pkl",
            OccurrenceLogEntry(batch_idx=iteration, object=start_entries),
        )

        seeds = rng.integers(low=0, high=2**31, size=len(start_entries))
        rngs = list(map(np.random.default_rng, seeds))

        work_unit_count = hparams.go_explore_num_processes * 2
        start_entries_chunked = _chunk_list(start_entries, work_unit_count)
        rngs_chunked = _chunk_list(rngs, work_unit_count)

        _collect_rollouts = functools.partial(
            rollout,
            rollout_params,
            cache,
        )

        # TODO: Delete this print after debugging is over.
        print(f"We have this many entries: {len(cache._cache)}")
        with multiprocessing.Pool(processes=hparams.go_explore_num_processes) as pool:

            for rollout_success_entries, rollout_cache in pool.starmap(
                _collect_rollouts,
                [*zip(start_entries_chunked, rngs_chunked)],
            ):
                # TODO (Joseph): Figure out how to update the cache with the new cache correctly. Why am I updating the success entries and the cache separately?
                success_entries.update(rollout_success_entries)
                cache.update(rollout_cache)

    object_logger.log(
        f"state_cache-{hparams.env_width}.pkl",
        OccurrenceLogEntry(batch_idx=0, object=cache),
    )

    return success_entries


if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    parser = config.get_hyperparam_parser(
        config.bridger_config,
        description="Hyperparameter Parser for the BridgeBuilderModel",
        parser=None,
    )
    hparams = parser.parse_args()

    with ObjectLogManager(
        "object_logging", "go_explore", create_experiment_dir=True
    ) as object_logger:
        success_entries = explore(
            object_logger=object_logger,
            hparams=hparams,
        )

        object_logger.log("success_entry.pkl", success_entries)

        print(
            f"==========\nEntry Count: {len(success_entries)}\n * wa-sampled: {hparams.go_explore_wa_sampled}\n * wa-new: {hparams.go_explore_wa_led_to_something_new}\n * wa-visit: {hparams.go_explore_wa_times_visited}\nShortest: {sorted([len(x.trajectory) for x in success_entries ])}"
        )
