"""Tests for the action_inversion_checker tool."""
import unittest

import copy
import gym
import pathlib
import itertools
import numpy as np

from parameterized import parameterized

import torch

from typing import List, Optional

from bridger.debug import action_inversion_checker
from bridger import builder_trainer
from bridger import policies
from bridger import 

_ENV_NAME = "gym_bridges.envs:Bridges-v0"


def _constant_estimators(nA: int) -> List:
    return [
        (
            lambda action: lambda state: torch.nn.functional.one_hot(
                torch.tensor(action), nA
            )
        )(action)
        for action in range(nA)
    ]


class ActionInversionCheckerTest(unittest.TestCase):
    def setUp(self):
        self._env = builder_trainer.make_env(
            name=_ENV_NAME, width=6, force_standard_config=True
        )
        self._constant_estimators = _constant_estimators(self._env.nA)
        self._policies = [
            policies.GreedyPolicy(constant_estimator)
            for constant_estimator in self._constant_estimators
        ]

    @parameterized.expand(
        [
            ("0", 0),
            ("1", 1),
            ("2", 2),
            ("3", 3),
            ("4", 4),
            ("5", 5),
        ]
    )
    def test_check_unknown_state(self, name: str, policy_index: int):
        """Verifies a state that should not be visited does not report inversion."""
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(
            env=self._env, actions=actions
        )

        self._env.reset()
        self._env.step(0)
        state, _, _, _ = self._env.step(0)

        self.assertEqual(
            checker.check(policy=self._policies[policy_index], states=[state]), []
        )

    def test_check_converged_and_preferred(self):
        """Verifies management of converged and preferred states."""
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(
            env=self._env, actions=actions
        )

        state = self._env.reset()

        states = [state, state]

        # Attempt non-preferred action in non-converged state.
        self.assertEqual(checker.check(policy=self._policies[1], states=states), [])
        self.assertEqual(checker.check(policy=self._policies[1], states=states), [])

        # Converge the state.
        self.assertEqual(checker.check(policy=self._policies[0], states=states), [])

        # Attempt a non-preferred action in a converged
        # state. Duplicating the state verifies that each state is
        # checked and reported on separately.
        expected_report = action_inversion_checker.ActionInversionReport(
            state=state, preferred_actions={0, 4}, policy_action=1
        )
        self.assertEqual(
            checker.check(policy=self._policies[1], states=states),
            [expected_report, expected_report],
        )

        # Attempt varying preferred actions in a converged state.
        self.assertEqual(checker.check(policy=self._policies[0], states=states), [])
        self.assertEqual(checker.check(policy=self._policies[4], states=states), [])

    @parameterized.expand(
        [
            ("Single column", [[0, 0, 0], [4, 4, 4]]),
            ("Single column raised", [[0, 0, 0, 0], [4, 4, 4, 4]]),
            ("Single column more raised", [[0, 0, 0, 0, 0], [4, 4, 4, 4, 4]]),
        ]
    )
    def test_check_actions_parsing(self, name: str, actions: List[List[int]]):
        """Verifies that different action inputs are parsed differently and correctly."""
        checker = action_inversion_checker.ActionInversionChecker(
            env=self._env, actions=actions
        )

        state = self._env.reset()
        states = [state]
        expected_reports = []

        for i in range(len(actions[0])):
            # No error for the newest state. The older states still have errors.
            self.assertEqual(
                checker.check(policy=self._policies[1], states=states), expected_reports
            )
            # Converge the state.
            self.assertEqual(checker.check(policy=self._policies[0], states=states), [])
            # Move is no longer ok for any state in states.
            expected_reports.append(
                action_inversion_checker.ActionInversionReport(
                    state=state, preferred_actions={0, 4}, policy_action=1
                )
            )
            self.assertEqual(
                checker.check(policy=self._policies[1], states=states), expected_reports
            )
            state, _, _, _ = self._env.step(0)
            states.append(state)

        # Verify that after the last block in the 0 column, the next
        # block placement does not result in a change in the checker.
        self.assertEqual(
            checker.check(policy=self._policies[1], states=states), expected_reports
        )
        self.assertEqual(checker.check(policy=self._policies[0], states=states), [])
        self.assertEqual(
            checker.check(policy=self._policies[1], states=states), expected_reports
        )

    def test_check_preferred_actions_ignore_bad_actions(self):
        """Verifies preferred actions do not include actions that don't change the env."""
        actions = [[0, 0], [3, 3]]
        checker = action_inversion_checker.ActionInversionChecker(
            env=self._env, actions=actions
        )

        state = self._env.reset()
        self.assertEqual(checker.check(policy=self._policies[0], states=[state]), [])
        self.assertEqual(
            checker.check(policy=self._policies[1], states=[state]),
            [
                action_inversion_checker.ActionInversionReport(
                    state=state, preferred_actions={0}, policy_action=1
                )
            ],
        )

    def test_check_illegal_actions(self):
        actions = [[0, 1]]
        self.assertRaisesRegex(
            ValueError,
            "Actions must contain exactly 2",
            action_inversion_checker.ActionInversionChecker,
            env=self._env,
            actions=actions,
        )

        actions = [[0, 1], [4, 3], [4, 3]]
        self.assertRaisesRegex(
            ValueError,
            "Actions must contain exactly 2",
            action_inversion_checker.ActionInversionChecker,
            env=self._env,
            actions=actions,
        )

    @parameterized.expand(
        [
            ("Always 0", 0, 5),
            ("Always 1", 1, 5),
            ("Always 2", 2, 8),
            ("Always 3", 3, 5),
            ("Always 4", 4, 5),
            ("Always 5", 5, 8),
        ]
    )
    def test_check_using_all_bridge_states(
        self, name: str, policy_index: int, expected_inversion_count: int
    ):
        """Verifies that checking all the states involved in building the bridge.

        Actions 2 and 5 are never correct and result in an inversion for every state (8).

        The remaining actions are each valid for 3 of 8 states.
        """
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(
            env=self._env, actions=actions
        )

        # Converge every state.
        for policy in self._policies:
            checker.check(policy=policy)

        # Check the inversions for the current policy.
        reports = checker.check(policy=self._policies[policy_index])
        self.assertEqual(            len(report),            expected_inversion_count        )
        for report in reports:
            self.assertEqual(report.policy_action, policy_index)


if __name__ == "__main__":
    unittest.main()
