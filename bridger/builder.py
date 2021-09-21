"""The Builder completes construction projects.

A Builder will run construction episodes within a given environment
using the provided policy.

  Typical Usage:

  builder = Builder(env)
  build_result = builder.build(policy, episode_length=40)
"""

import dataclasses
import gym
import numpy as np
import torch
from typing import Any

from bridger import policies


class BuildEvaluator:
    """The BuildEvaluator scores the quality of a building policy.

    The BuildEvaluator executes the requested number of builds using a
    provided policy and calculates a series of stats to rate the
    policy. The goal is to provide both an absolute notion of the
    building policy as well as enable relative comparisons between
    policies.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: policies.Policy,
        build_count: int,
        episode_length: int,
    ):
        self._env = env
        self._build_count = build_count
        self._episode_length = episode_length
        builder = Builder(env)
        self._build_results = [
            builder.build(
                policy=policy, episode_length=self._episode_length, render=False
            )
            for _ in range(self._build_count)
        ]

    @property
    def success_rate(self):
        """Returns the rate of successful builds vs build attempts."""
        return self.successes.mean()

    @property
    def successes(self):
        """Returns whether the build was successful for all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return np.array([build_result.success for build_result in self._build_results])

    @property
    def build_steps_on_success_mean(self):
        """Returns the mean of build steps taken on successful builds only."""
        return self.build_steps.mean(where=self.successes)

    @property
    def build_steps(self):
        """Returns the build steps taken from all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return np.array([build_result.steps for build_result in self._build_results])

    @property
    def reward_on_success_mean(self):
        """Returns the mean of build steps taken on successful builds only."""
        return self.rewards.mean(where=self.successes)

    @property
    def rewards(self):
        """Returns rewards from all episodes in episode order.

        Returning the full ordered list enables comparing success and
        failure points for different policies.

        """
        return np.array([build_result.reward for build_result in self._build_results])

    @property
    def height_of_highest_block_mean(self):
        """Demo metric using the final state.

        Computes the height of the highest brick. This metric should
        be replaced as we find more interesting ones to evaluate the
        constructed bridge.

        """
        # The following computes whether each row of the final_state
        # contains a brick and then selects the lowest state row which
        # contains a brick. In the inverted representation, a lower
        # row in final_state corresponds to the higher row in the
        # rendered env. The inversion to translate to bridge height is
        # computed before returning.
        end_states = np.stack(
            [build_result.final_state for build_result in self._build_results]
        )
        inverted_heights = (end_states == self._env.StateType.BRICK).any(-1).argmax(-1)
        return end_states.shape[1] - 1 - inverted_heights.mean()

    def print_report(self):
        print(
            "Build Evaluation Summary\n"
            f"{self._build_count} build episodes of up to {self._episode_length} "
            "steps each.\n"
            f"Success rate: {self.success_rate:.2f}\n"
            f"Mean height of highest block: {self.height_of_highest_block_mean:.2f}\n"
            f"On Success:\n"
            f"  Mean Rewards: {self.reward_on_success_mean:.2f}\n"
            f"  Build Steps: {self.build_steps_on_success_mean:.2f}"
        )


# pylint: disable=missing-class-docstring
@dataclasses.dataclass
class BuildResult:
    success: bool
    reward: float
    steps: int
    final_state: Any


# pylint: disable=too-few-public-methods
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
            # pylint: disable=not-callable
            state, reward, success, _ = self._env.step(policy(torch.tensor(state)))
            total_reward += reward
            if render:
                self._env.render()
            if success:
                break

        return BuildResult(
            success=success, reward=total_reward, steps=i + 1, final_state=state
        )
