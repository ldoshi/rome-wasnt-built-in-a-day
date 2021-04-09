import numpy as np
import unittest
from parameterized import parameterized

from bridger import policies


def _constant_estimator(state):
    return np.array([1, 0, 0, 0, 0])


def _state_is_action_estimator(state):
    q_values = np.zeros(5)
    q_values[state % 5] = 1
    return q_values


def _noisy_state_is_action_estimator(state):
    q_values = np.random.rand(5)
    q_values[state % 5] = 1
    return q_values


class TestProbabilities(unittest.TestCase):
    @parameterized.expand(
        [
            (_constant_estimator, state, epsilon, 0)
            for state in range(5)
            for epsilon in np.linspace(0, 1, 10)
        ]
        + [
            (estimator, state, epsilon, state % 5)
            for estimator in [
                _state_is_action_estimator,
                _noisy_state_is_action_estimator,
            ]
            for state in range(10)
            for epsilon in np.linspace(0, 1, 10)
        ]
    )
    def test_eps_greedy_policy(self, estimator, state, epsilon, mode_action):
        policy = policies.EpsilonGreedyPolicy(estimator, epsilon=epsilon)
        probs = policy.get_probabilities(state)
        self.assertAlmostEqual(probs[mode_action - 1], epsilon / len(probs))
        if epsilon != 1:
            self.assertEqual(len(set(probs)), 2)
            self.assertEqual(probs.argmax(), mode_action)
        self.assertAlmostEqual(probs.sum(), 1)

    @parameterized.expand(
        [(_constant_estimator, state, 0) for state in range(5)]
        + [
            (estimator, state, state % 5)
            for estimator in [
                _state_is_action_estimator,
                _noisy_state_is_action_estimator,
            ]
            for state in range(10)
        ]
    )
    def test_greedy_policy(self, estimator, state, mode_action):
        policy = policies.GreedyPolicy(estimator)
        probs = policy.get_probabilities(state)
        expected = np.zeros(probs.shape)
        expected[mode_action] = 1
        self.assertTrue(np.allclose(probs, expected))


if __name__ == "__main__":
    unittest.main()
