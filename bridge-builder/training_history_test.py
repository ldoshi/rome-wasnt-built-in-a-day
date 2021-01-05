import numpy as np
import unittest
from training_history import TrainingHistory


class TestTrainingHistory(unittest.TestCase):

    def setUp(self):
        self.history = TrainingHistory()
        self.states = [np.array([[0,0],[0,0]]),
                       np.array([[1,0],[1,1]]),
                       np.array([[0,1],[1,0]])]

        self.history.increment_visit_count(self.states[0])
        self.history.increment_visit_count(self.states[0])
        self.history.increment_visit_count(self.states[2])
        self.history.increment_visit_count(self.states[1])
        self.history.increment_visit_count(self.states[1])
        self.history.increment_visit_count(self.states[1])

        self.base = np.array([1,2,3])
        self.history.add_q_values(self.states[0], 1, self.base, self.base * 2)
        self.history.add_q_values(self.states[0], 10, self.base*10, self.base * 10 * 2)
        self.history.add_q_values(self.states[1], 1, self.base + 1, (self.base+1) * 2)
        self.history.add_q_values(self.states[1], 10, (self.base + 1)*10, (self.base+1) * 10 * 2)
        self.history.add_q_values(self.states[2], 1, self.base + 2, (self.base+2) * 2)
        self.history.add_q_values(self.states[2], 10, (self.base + 2)*10, (self.base+2) * 10 * 2)

        self.history.add_td_target_delta(self.states[0], 1, 1, 10)
        self.history.add_td_target_delta(self.states[0], 1, 10, 5)
        self.history.add_td_target_delta(self.states[0], 2, 11, 6)
        self.history.add_td_target_delta(self.states[2], 1, 1, 5)

    def test_verify_data_series(self):
        # Use one of the access methods, though we are not testing details here.
        test_history = self.history.get_history_by_visit_count()

        test_map = {}
        for x in test_history:
            test_map[str(x.state)] = x

        # Validate q and q target values.
        for i in range(len(self.states)):
            state_training_history = test_map[str(self.states[i])]
            for a in range(len(self.base)):
                epochs, values = state_training_history.get_q_values(a)
                self.assertListEqual(epochs, [1, 10])
                self.assertListEqual(values, [self.base[a] + i, (self.base[a] + i ) * 10])

                epochs, values = state_training_history.get_q_target_values(a)
                self.assertListEqual(epochs, [1, 10])
                self.assertListEqual(values, [(self.base[a] + i) * 2 , (self.base[a] + i ) * 10 * 2])
            
        # Validate td target deltas.
        state_training_history = test_map[str(self.states[0])]
        epochs, values = state_training_history.get_td_target_deltas(1)
        self.assertListEqual(epochs, [1, 10])
        self.assertListEqual(values, [10, 5])       
        epochs, values = state_training_history.get_td_target_deltas(2)
        self.assertListEqual(epochs, [11])
        self.assertListEqual(values, [6])
        epochs, values = state_training_history.get_td_target_deltas(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

        state_training_history = test_map[str(self.states[1])]
        epochs, values = state_training_history.get_td_target_deltas(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

        state_training_history = test_map[str(self.states[2])]
        epochs, values = state_training_history.get_td_target_deltas(1)
        self.assertListEqual(epochs, [1])
        self.assertListEqual(values, [5])
        epochs, values = state_training_history.get_td_target_deltas(0)
        self.assertListEqual(epochs, [])
        self.assertListEqual(values, [])

        
    def test_get_history_by_vist_count(self):
        test_history = self.history.get_history_by_visit_count()
        expected = [self.states[1], self.states[0], self.states[2]]
        for i,x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

        test_history = self.history.get_history_by_visit_count(n=2)
        expected = [self.states[1], self.states[0]]
        for i,x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

    def test_get_history_by_state_complexity(self):
        test_history = self.history.get_history_by_complexity()
        expected = [self.states[0], self.states[2], self.states[1]]
        for i,x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

        test_history = self.history.get_history_by_complexity(n=2)
        expected = [self.states[0], self.states[2]]
        for i,x in enumerate(test_history):
            self.assertTrue(np.array_equal(x.state, expected[i]))

if __name__ == '__main__':
    unittest.main()
