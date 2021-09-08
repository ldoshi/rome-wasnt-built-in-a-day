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


#pylint: disable=missing-class-docstring
@dataclasses.dataclass
class BuildResult:
    success: bool
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
          than episode_length if the construction is successful.
        """
        state = self._env.reset()
        total_reward = 0
        for i in range(episode_length):
            # This lint error seems to be a torch+pylint issue in general.
            #pylint: disable=not-callable
            state, reward, success, _ = self._env.step(policy(torch.tensor(state)))
            total_reward += reward
            if render:
                self._env.render()
            if success:
                break

        return BuildResult(success=success, reward=total_reward, steps=i + 1)
