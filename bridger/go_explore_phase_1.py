from gym_bridges.envs.bridges_env import BridgesEnv
import copy
from typing import Any
from bridger import    hash_utils
from dataclasses import dataclass
import numpy as np

@dataclass
class CacheEntry:
    trajectory: list[int]
    reward: float
    steps_since_led_to_something_new: int = 0
    sampled_count: int = 0
    visit_count: int = 1

@dataclass
class SuccessEntry:
    trajectory: list[int]
    reward: float    

class StateCache:

    _cache: dict[Any, CacheEntry] = {}

    def __init__(self, rng):
        self._rng = rng

    def update_times_since_led_to_something_new(self, state, led_to_something_to_new: bool) -> None:
        key = hash_utils.hash_tensor(state)
        assert key in self._cache
        if led_to_something_to_new:
            self._cache[key].steps_since_led_to_something_new = 0
        else:
            self._cache[key].steps_since_led_to_something_new += 1
    
    def visit(self, state, trajectory: list[int], reward: float) -> bool:
        key = hash_utils.hash_tensor(state)
        if key in self._cache:
            entry = self._cache[key]
            entry.visit_count += 1
            if reward > entry.reward or (reward == entry.reward and len(trajectory) < len(entry.trajectory)):
                entry.reward = reward
                entry.trajectory = trajectory
        else:
            self._cache[key] = CacheEntry(trajectory=trajectory, reward=reward)

    def sample(self, n=1):
        # Fix this when n > 1.
        assert n == 1
        cache_keys = list(self._cache)
        key_index = self._rng.choice(range(len(cache_keys)), n)[0]
        key = cache_keys[key_index]
        entry = self._cache[key]
        entry.sampled_count += 1
        return (np.array(key[1]).reshape(key[0]), entry)
        

def rollout(rng, env, start_state, start_entry, cache, num_actions, success_entries):
    env.reset(start_state)

    current_trajectory = copy.deepcopy(start_entry.trajectory)
    current_reward = start_entry.reward

    led_to_something_new = False
    for _ in range(num_actions):
        action = rng.choice(range(env.nA))
        current_trajectory.append(action)
        next_state, reward, done, _ = env.step(action)
        current_reward += reward
        if done:
            success_entries.append(SuccessEntry(trajectory=current_trajectory, reward=current_reward))
            led_to_something_new = True
            break

        cache.visit(next_state, current_trajectory, current_reward)

    cache.update_times_since_led_to_something_new(start_state, led_to_something_new)
    
def explore(rng, env, cache, num_iterations, num_actions, success_entries):

    for _ in range(num_iterations):
        start_state, start_entry = cache.sample()
        rollout(rng, env, start_state, start_entry, cache, num_actions, success_entries)
        
    
rng = np.random.default_rng(42)
width=6
env = BridgesEnv(width=width, force_standard_config=True)

num_iterations = 10000
num_actions = 8

cache: StateCache = StateCache(rng)
cache.visit(state=env.reset(), trajectory=[], reward=0)
success_entries: list[SuccessEntry] = []

explore(rng, env, cache, num_iterations, num_actions, success_entries)
