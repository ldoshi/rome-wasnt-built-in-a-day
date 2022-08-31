"""The ActionInversionChecker signals when unexpected actions gain priority.

The ActionInversionChecker specifically signals when the q-value of an
unexpected action rises enough to make it the preferred action under a
deterministic policy. This can be used to pinpoint when the model
starts unexpectedly deteriorating during training.

This tool is intended for debugging purposes only.

  Typical Usage:

"""

from collections.abc import Hashable

import dataclasses
import gym
import numpy as np
import torch
from typing import List, Callable

from bridger import policies


@dataclasses.dataclass
class ActionInversionReport:
    state: torch.Tensor
    preferred_actions: List[int]
    policy_action: int


class ActionInversionChecker:
    """The"""

    def __init__(
        self,
        env: gym.Env,
        actions: List[List[int]],
        # TODO(lyric): Swap in the new one.
        state_hash_fn: Callable[[torch.Tensor], Hashable] = str,
    ):
        """Initialize expected actions based on the provided contruction sequence.

        The construction sequence is used to determine preferred
        action candidates for each intermediate state attained via all
        valid permutations of the construction sequence.

        This implementation presumes an environment where the bridge
        is built up from the left and the right without an
        intermediate supports.

        Args:

          env: A gym for simulating construction. The current
            implementation assumes that reset() allows returns the same
            starting state.
          actions: The sequence of construction actions defining the
            target outcome. We expect actions to be populated with
            exactly two lists. The first contains actions for the left
            side of the bridge. The second contains actions for the
            right side of the bridge. Each list contains actions that
            will place blocks from the ground up until that half of the
            bridge is complete.
          state_hash_fn: A function that converts states into something
            that can be hashed.

        Raises:
          ValueError if actions does not have the correct format.

        """
        self._state_hash_fn = state_hash_fn
        if len(actions) != 2:
            raise ValueError(
                f"Actions must contain exactly 2 elements, provided {len(actions)}"
            )

        # Empirically check that env.reset() seems to consistently
        # return the same starting state.
        starting_state = env.reset()
        for _ in range(10):
            new_starting_state = env.reset()
            if (starting_state != new_starting_state).any():
                raise ValueError(
                    "The env.reset() does not return the same starting state."
                    f"For example,\n{starting_state}\nvs\n{new_starting_state}"
                )

        self._preferred_actions = {}
        self._populate_preferred_actions(
            env=env,
            actions=actions,
            action_indices=[0] * len(actions),
            state=env.reset(),
        )
        # Contains states which have converged to align with the
        # ActionInversionChecker. Once a state converges, it should
        # not diverge.
        self._converged_states = set()

    def _populate_preferred_actions(
        self,
        env: gym.Env,
        actions: List[List[int]],
        action_indices: List[int],
        state: List[torch.Tensor],
    ) -> None:
        preferred_actions = []
        for index, (action_list, action_index) in enumerate(
            zip(actions, action_indices)
        ):
            if action_index < len(action_list):
                preferred_actions.append((index, action_list[action_index]))

        if preferred_actions:
            self._preferred_actions[self._state_hash_fn(state)] = set(
                [element[1] for element in preferred_actions]
            )

            for index, action in preferred_actions:
                env.reset(state)
                next_state, _, _, _ = env.step(action)
                action_indices[index] += 1
                self._populate_preferred_actions(
                    env, actions, action_indices, next_state
                )
                action_indices[index] -= 1

    def check(
        self, policy: policies.Policy, states: List[torch.Tensor]
    ) -> List[ActionInversionReport]:
        """

        Args:
          policy: The policy used to determine each next brick placement.

        """

        action_inversions = []
        for state in states:
            state_hash = self._state_hash_fn(state)

            if state_hash not in self._preferred_actions:
                continue

            preferred_actions = self._preferred_actions[state_hash]
            policy_action = policy(state)

            if (
                policy_action not in preferred_actions
                and state_hash in self._converged_states
            ):
                action_inversions.append(
                    ActionInversionReport(
                        state=state,
                        preferred_actions=preferred_actions,
                        policy_action=policy_action,
                    )
                )
            if (
                policy_action in preferred_actions
                and state_hash not in self._converged_states
            ):
                self._converged_states.add(state_hash)

        #        in theory, want to cover all states we have here. but...
        #        need support for two things: Don't worry about states they are not visited yet. this is subsumed by dont' worry about states for which we havent converged to best action yet. once converged, not allowed to leave. so keep a set() of states that are expected to match!

        # Either in this class or the caller, consider options like a) allowed n inversion before flagged or b) wait until batch_idx = n before enforcing.

        # Verifies that all states do not have an inversion from policy (or complain aptly?)
        return action_inversions
