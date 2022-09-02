"""The ActionInversionChecker signals when unexpected actions gain priority.

The ActionInversionChecker specifically signals when the q-value of an
unexpected action rises enough to make it the preferred action under a
deterministic policy. This can be used to pinpoint when the model
starts unexpectedly deteriorating during training.

This tool is intended for debugging purposes only.

Usage:
  actions = [[0, 1], [4, 3]]
  checker = ActionInversionChecker(env=env, actions=actions)

  for each training step:
    <update policy>
    report = checker.check(policy)

"""

from collections.abc import Hashable

import dataclasses
import gym
import numpy as np
import torch
from typing import Callable, List, Optional, Set

from bridger import hash_utils
from bridger import policies


@dataclasses.dataclass
class PreferredActionEntry:
    """A state paired with the preferred actions to take from that state."""

    state: torch.Tensor
    preferred_actions: Set[int]


@dataclasses.dataclass
class ActionInversionReport:
    """An incidence of action inversion for a particular state."""

    state: torch.Tensor
    preferred_actions: Set[int]
    policy_action: int


class ActionInversionChecker:
    """The action inversion checker checks for suboptimal policy decisions.

    The class maintains state across calls to check. Once the policy
    converges to a preferred action for a state, it is expected to
    always recommend a preferred action. Failing to do so results in
    an ActionInversionReport for that state.

    """

    def __init__(
        self,
        env: gym.Env,
        actions: List[List[int]],
        state_hash_fn: Callable[[torch.Tensor], Hashable] = hash_utils.hash_tensor,
    ):
        """Initialize expected actions based on the provided contruction sequence.

        The construction sequence is used to determine preferred
        action candidates for each intermediate state attained via all
        valid permutations of the construction sequence.

        This implementation presumes an environment where the bridge
        is built up from edge to edge without intermediate supports.

        Args:
          env: A gym for simulating construction. The current
            implementation assumes that reset() always returns the same
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

        self._states_to_preferred_actions = {}
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
        """Populates the preferred actions per state.

        This function uses recursive calls to enumerate all states
        generated by the permutations of actions from the left and
        right. The preferred actions to take from each enumerated
        state are stored into an internal data structure.

        Args:
          env: A gym for simulating construction. The current
            implementation assumes that reset() always returns the same
            starting state.
          actions: The sequence of construction actions defining the
            target outcome. We expect actions to be populated with
            exactly two lists. The first contains actions for the left
            side of the bridge. The second contains actions for the
            right side of the bridge. Each list contains actions that
            will place blocks from the ground up until that half of the
            bridge is complete.
          action_indices: A list of indices identifying the current
            action to take for each sublist of actions.
          state: The state for which to determine preferred actions.
        """

        preferred_actions = []
        for index, (action_list, action_index) in enumerate(
            zip(actions, action_indices)
        ):
            if action_index < len(action_list):
                preferred_actions.append((index, action_list[action_index]))

        if preferred_actions:
            self._states_to_preferred_actions[
                self._state_hash_fn(state)
            ] = PreferredActionEntry(
                state=state,
                preferred_actions=set([element[1] for element in preferred_actions]),
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
        self, policy: policies.Policy, states: Optional[List[torch.Tensor]] = None
    ) -> List[ActionInversionReport]:
        """Checks the policy for action inversions.

        Args:
          policy: The policy used to determine each next brick placement.
          states: The list of states for which to check the policy. If
            None, then the check is run against all states encountered
            while building the bridge defined by actions in the __init__.

        Returns:
          A list of ActionInversionReport with a report for each
          incident where the policy does not identify a preferred action
          for a converged state.
        """

        action_inversions = []

        if states is None:
            states = [
                entry.state for entry in self._states_to_preferred_actions.values()
            ]

        for state in states:
            state_hash = self._state_hash_fn(state)

            if state_hash not in self._states_to_preferred_actions:
                continue

            preferred_actions = self._states_to_preferred_actions[
                state_hash
            ].preferred_actions
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

        return action_inversions
