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

_ENV_NAME = "gym_bridges.envs:Bridges-v0"
_DELTA = 1e-6

def _constant_estimators(nA: int) -> List:
    return [(lambda action: lambda state: torch.nn.functional.one_hot(torch.tensor(action), nA))(action) for action in range(nA)]
    
class ActionInversionCheckerTest(unittest.TestCase):

    # @parameterized.expand(
    #     [
    #         ("No Early Stopping", []),
    #         (
    #             "Early Stopping",
    #             [
    #                 EarlyStopping(
    #                     monitor="val_reward",
    #                     patience=1,
    #                     mode="max",
    #                     strict=True,
    #                     check_on_train_epoch_end=False,
    #                 )
    #             ],
    #         ),
    #     ]
    # )

    def test_check_unknown_state(self):
        """Verifies a state that should not be visited does not report inversion."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=6, force_standard_config=True
        )
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(env=env,actions=actions)

        env.reset()
        env.step(0)
        state, _, _, _ = env.step(0)

        constant_estimators = _constant_estimators(env.nA)
        for constant_estimator in constant_estimators:
            self.assertEqual(checker.check(states=[state],policy=policies.GreedyPolicy(constant_estimator)), [])
        
    def test_check_converged_and_preferred(self):
        """Verifies management of converged and preferred states."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=6, force_standard_config=True
        )
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(env=env,actions=actions)

        state = env.reset()

        constant_estimators = _constant_estimators(env.nA)
        states = [state, state]
        
        # Attempt non-preferred action in non-converged state.       
        self.assertEqual(checker.check(states=states, policy=policies.GreedyPolicy(constant_estimators[1])), [])
        self.assertEqual(checker.check(states=states, policy=policies.GreedyPolicy(constant_estimators[1])), [])

        # Converge the state.
        self.assertEqual(checker.check(states=states, policy=policies.GreedyPolicy(constant_estimators[0])), [])

        # Attempt a non-preferred action in a converged
        # state. Duplicating the state verifies that each state is
        # checked and reported on separately.
        expected_report = action_inversion_checker.ActionInversionReport(state=state, preferred_actions={0,4}, policy_action=1)
        self.assertEqual(checker.check(states=states, policy=policies.GreedyPolicy(constant_estimators[1])), [expected_report, expected_report])

        # Attempt a preferred action in a converged state.
        self.assertEqual(checker.check(states=state, policy=policies.GreedyPolicy(constant_estimators[4])), [])

#        then go piece by piece in check for if conditoins.
#        THEN, one test to ensure actions is parsed correctly into checker for fancier actions re: preferred actions. that can be independent of how check managers eg converged state 

    def atest_check(self):
        """Verifies that check manages preferred actions and converged states correctly."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=6, force_standard_config=True
        )
        actions = [[0, 1], [4, 3]]
        checker = action_inversion_checker.ActionInversionChecker(env=env,actions=actions)

        # Check s
        
        env.reset()
        env.step(0)
        state, _, _, _ = env.step(0)

        def _constant_estimator(state) -> torch.Tensor:
            """Returns a policy that always adds a brick to the left side."""
            return torch.tensor([1, 0, 0, 0])
        
        self.assertEqual(checker.check(states=[state],policy=policies.GreedyPolicy(_constant_estimator)), [])
        
        
        


if __name__ == "__main__":
    unittest.main()
