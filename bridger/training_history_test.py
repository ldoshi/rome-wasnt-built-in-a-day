import numpy as np
import pathlib
import unittest
from bridger.training_history import TrainingHistory


def build_test_history(states=None, base=None, serialization_dir=None):
    if states is None:
        states = [
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            np.array([[0, 1], [1, 0]]),
        ]

    if base is None:
        base = np.array([1, 2, 3])

    history = TrainingHistory(serialization_dir=serialization_dir)

    history.increment_visit_count(states[0])
    history.increment_visit_count(states[0])
    history.increment_visit_count(states[2])
    history.increment_visit_count(states[1])
    history.increment_visit_count(states[1])
    history.increment_visit_count(states[1])

    history.add_q_values(1, states[0], base, base * 2)
    history.add_q_values(10, states[0], base * 10, base * 10 * 2)
    history.add_q_values(1, states[1], base + 1, (base + 1) * 2)
    history.add_q_values(10, states[1], (base + 1) * 10, (base + 1) * 10 * 2)
    history.add_q_values(1, states[2], base + 2, (base + 2) * 2)
    history.add_q_values(10, states[2], (base + 2) * 10, (base + 2) * 10 * 2)

    history.add_td_error(1, states[0], 1, 10)
    history.add_td_error(10, states[0], 1, 5)
    history.add_td_error(11, states[0], 2, 6)
    history.add_td_error(1, states[2], 1, 5)
    return history


_TMP_DIR = "tmp"


def clean_up_dir(path):
    for filepath in path.iterdir():
        filepath.unlink()


def create_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    clean_up_dir(path)


def delete_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    clean_up_dir(path)
    path.rmdir()


def is_history_equal(expected, actual):
    """A lightweight equality check that only compares visit counts. This
    will catch fundamental mismatches or fails in
    serialization-deserialization but not subtle bugs.

    """
    if len(expected._training_history) != len(actual._training_history):
        return False

    for state in expected._training_history:
        if (
            expected._training_history[state].visit_count
            != actual._training_history[state].visit_count
        ):
            return False

    return True


class TestTrainingHistory(unittest.TestCase):
    def setUp(self):
        self._states = [
            np.array([[0, 0], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            np.array([[0, 1], [1, 0]]),
        ]
        self._base = np.array([1, 2, 3])
        self._history = build_test_history(
            states=self._states, base=self._base, serialization_dir=_TMP_DIR
        )
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()

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

    def test_serialize_deserialize(self):
        new_history = TrainingHistory(deserialization_dir=_TMP_DIR)
        self.assertFalse(is_history_equal(self._history, new_history))
        # Nothing to deserialize yet.
        self.assertFalse(new_history.deserialize_latest())

        # The new_history should now match _history.
        self._history.serialize()
        self.assertTrue(new_history.deserialize_latest())
        self.assertTrue(is_history_equal(self._history, new_history))

        # No new changes to new_history. It should still match _history.
        self.assertFalse(new_history.deserialize_latest())
        self.assertTrue(is_history_equal(self._history, new_history))

        # Add a change to _history. The new_history no longer matches
        # until we re-serialize and deserialize.
        self._history.increment_visit_count(self._states[0])
        self.assertFalse(is_history_equal(self._history, new_history))
        self.assertFalse(new_history.deserialize_latest())
        self._history.serialize()
        self.assertTrue(new_history.deserialize_latest())
        self.assertTrue(is_history_equal(self._history, new_history))


if __name__ == "__main__":
    unittest.main()
