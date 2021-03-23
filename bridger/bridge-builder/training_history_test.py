import numpy as np
import unittest
from training_history import TrainingHistory


def build_test_history(states=None, base=None):
    if states is None:
        states = [
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            np.array([[0, 1], [1, 0]]),
        ]

    if base is None:
        base = np.array([1, 2, 3])

    history = TrainingHistory()

    history.increment_visit_count(states[0])
    history.increment_visit_count(states[0])
    history.increment_visit_count(states[2])
    history.increment_visit_count(states[1])
    history.increment_visit_count(states[1])
    history.increment_visit_count(states[1])

    history.add_q_values(states[0], 1, base, base * 2)
    history.add_q_values(states[0], 10, base * 10, base * 10 * 2)
    history.add_q_values(states[1], 1, base + 1, (base + 1) * 2)
    history.add_q_values(states[1], 10, (base + 1) * 10, (base + 1) * 10 * 2)
    history.add_q_values(states[2], 1, base + 2, (base + 2) * 2)
    history.add_q_values(states[2], 10, (base + 2) * 10, (base + 2) * 10 * 2)

    history.add_td_error(states[0], 1, 1, 10)
    history.add_td_error(states[0], 1, 10, 5)
    history.add_td_error(states[0], 2, 11, 6)
    history.add_td_error(states[2], 1, 1, 5)
    return history


class TestTrainingHistory(unittest.TestCase):
    def setUp(self):
        self._states = [
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            np.array([[0, 1], [1, 0]]),
        ]
        self._base = np.array([1, 2, 3])
        self._history = build_test_history(states=self._states, base=self._base)

    def test_verify_data_series(self):
        # Use one of the access methods, though we are not testing details here.
        test_history = self._history.get_history_by_visit_count()

        test_map = {}
        for x in test_history:
            test_map[str(x.state)] = x

        # Validate q and q target values.
        for i in range(len(self._states)):
            state_training_history = test_map[str(self._states[i])]
            for a in range(len(self._base)):
                epochs, values = state_training_history.get_q_values(a)
                self.assertListEqual(epochs, [1, 10])
                self.assertListEqual(
                    values, [self._base[a] + i, (self._base[a] + i) * 10]
                )

                epochs, values = state_training_history.get_q_target_values(a)
                self.assertListEqual(epochs, [1, 10])
                self.assertListEqual(
                    values, [(self._base[a] + i) * 2, (self._base[a] + i) * 10 * 2]
                )

        # Validate td target deltas.
        state_training_history = test_map[str(self._states[0])]
        epochs, values = state_training_history.get_td_errors(1)
        self.assertListEqual(epochs, [1, 10])
        self.assertListEqual(values, [10, 5])
        epochs, values = state_training_history.get_td_errors(2)
        self.assertListEqual(epochs, [11])
        self.assertListEqual(values, [6])
        epochs, values = state_training_history.get_td_errors(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

        state_training_history = test_map[str(self._states[1])]
        epochs, values = state_training_history.get_td_errors(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

        state_training_history = test_map[str(self._states[2])]
        epochs, values = state_training_history.get_td_errors(1)
        self.assertListEqual(epochs, [1])
        self.assertListEqual(values, [5])
        epochs, values = state_training_history.get_td_errors(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

    def test_get_history_by_vist_count(self):
        test_history = self._history.get_history_by_visit_count()
        expected = [self._states[1], self._states[0], self._states[2]]
        for i, x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

        test_history = self._history.get_history_by_visit_count(n=2)
        expected = [self._states[1], self._states[0]]
        for i, x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

    def test_get_history_by_state_complexity(self):
        test_history = self._history.get_history_by_complexity()
        expected = [self._states[0], self._states[2], self._states[1]]
        for i, x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

        test_history = self._history.get_history_by_complexity(n=2)
        expected = [self._states[0], self._states[2]]
        for i, x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))


if __name__ == "__main__":
    unittest.main()
