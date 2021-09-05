"""The Builder completes construction projects.

A Builder will run construction episodes within a given environment
using the provided policy.

  Typical Usage:

  builder = Builder(env)
  build_result = builder.build(policy, episode_length=40)
"""

import dataclasses
import gym
import torch

from bridger import policies

class BuildEvaluator:
    """The BuildEvaluator scores the quality of a building policy.

    The BuildEvaluator executes the requested number of builds using a
    provided policy and calculates a series of stats to rate the
    policy. The goal is to provide both an absolute notion of the
    building policy as well as enable relative comparisons between
    policies.
    """
    def __init__(self, env: gym.Env, policy: policies.Policy, build_count: int, episode_length: int):
        self._build_results = []
        builder = Builder(env)
        for _ in range(build_count):
            build_results.append(build(policy=policy, episode_length=episode_length, render=False))

    @property
    def success_rate(self):
        """Returns the rate of successful builds vs build attempts."""
        return np.sum(success_distribution) / len(self._build_results)

    @property
    def successes(self):
        """Returns whether the build was successful for all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return [build_result.success for build_result in self._build_results]

    @property
    def build_steps_on_success_mean(self):
        """Returns the mean of build steps taken on successful builds only."""
        return np.mean([build_result.steps for build_result in self._build_results if build_result.success])
        
    @property
    def build_steps(self):
        """Returns the build steps taken from all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return [build_result.steps for build_result in self._build_results]

    @property
    def reward_on_success_mean(self):
        """Returns the mean of build steps taken on successful builds only."""
        return np.mean([build_result.reward for build_result in self._build_results if build_result.success])
    
    @property
    def rewards(self):
        """Returns rewards from all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return [build_result.reward for build_result in self._build_results]

    add one more metric that invovles builder callback?
    

#pylint: disable=missing-class-docstring
@dataclasses.dataclass
class BuildResult:
    finished: bool
    reward: float
    steps: int

#pylint: disable=too-few-public-methods
class Builder:
    """The Builder supports repeated construction in the provided env."""

    def __init__(self, env: gym.Env):
        self._env = env

    def build(
        self, policy: policies.Policy, episode_length: int, render: bool = True
    ) -> BuildResult:
        """Builds following the provided policy.

        Resets the environment and constructs a fresh element
        following the provided policy.

        Args:
          policy: The policy used to determine each next brick placement.
          episode_length: The maximum number of steps to take.
          render: Whether to render the environment after each step.

        Returns:
          A BuildResult summarizing whether the construction was
          successful, the total reward for the construction, and how
          many steps were actually taken. The steps taken may be less
          episode_length if the construction is successful.
        """
        state = self._env.reset()
        total_reward = 0
        for i in range(episode_length):
            # This lint error seems to be a torch+pylint issue in general.
            #pylint: disable=not-callable
            state, reward, finished, _ = self._env.step(policy(torch.tensor(state)))
            total_reward += reward
            if render:
                self._env.render()
            if finished:
                break

        return BuildResult(finished=finished, reward=total_reward, steps=i + 1)

