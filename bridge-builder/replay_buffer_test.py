import numpy as np
import unittest
from replay_buffer import SumTree

class TestSumTree(unittest.TestCase):
    
    def setUp(self):
        self._delta = 1e-6
    
    def test_illegal_tree_usage(self):
        tree = SumTree(1)
        self.assertRaises(ValueError, tree.set_value, 1, 1)

    def test_small_tree_size(self):
        tree = SumTree(1)
        tree.set_value(0, 10)
        self.assertEqual(tree.max_value, 10)
        self.assertListEqual(tree.sample(0), [])
        self.assertListEqual(tree.sample(1), [(0, 1)])
        self.assertListEqual(tree.sample(3), [(0, 1), (0, 1), (0, 1)])

    def test_uniform_tree(self):
        capacity = 21
        tree = SumTree(capacity)
        boosted_priorities={}
        self._populate_sum_tree(tree, capacity, boosted_priorities)

        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=False)
        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=True)

        # Set the values from uniform trees to instead have some boosted priorities.
        boosted_priorities={1: 7, 19: 7}
        self._populate_sum_tree(tree, capacity, boosted_priorities)

        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=False)
        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=True)


    def test_prioritized_tree(self):
        capacity = 21
        tree = SumTree(capacity)
        boosted_priorities={1: 7, 19: 7}
        self._populate_sum_tree(tree, capacity, boosted_priorities)

        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=False)
        self._sample_and_verify(tree, capacity, boosted_priorities, stratified=True)
            
        # Set the values from boosted_priorities back to 1 and verify the tree is now uniform.
        self._populate_sum_tree(tree, capacity, boosted_priorities={}, max_value=tree.max_value)

        self._sample_and_verify(tree, capacity, boosted_priorities={}, stratified=False)
        self._sample_and_verify(tree, capacity, boosted_priorities={}, stratified=True)

    def _populate_sum_tree(self, tree, capacity, boosted_priorities, max_value=1):
        for i in range(capacity):
            if i in boosted_priorities:
                tree.set_value(i, boosted_priorities[i])
                max_value = max(max_value, boosted_priorities[i])
            else:
                tree.set_value(i, 1)
                
            self.assertEqual(tree.max_value, max_value)

    def _sample_and_verify(self, tree, capacity, boosted_priorities, stratified):
        samples = []
        iterations = 20000
        sample_count = 3

        for _ in range(iterations):
            sample = tree.sample(sample_count, stratified=stratified)
            self.assertEqual(len(sample), sample_count)
            samples.append(sample)

        sample_index_sets = [set() for _ in range(sample_count)]
        repeat_found = False

        lower_priority_numerator = (capacity - len(boosted_priorities))
        higher_priority_numerator = np.sum(np.array(list(boosted_priorities.values())))

        counts = np.zeros(capacity)

        for sample in samples:
            # If the sample is stratified, then each sample will never
            # repeat an index. Furthermore, the sample_count indices
            # will form sample_count disjoint sets when combined
            # across all iterations. For testing convenience, the test
            # cases are constructed to ensure no overlap instead of
            # little overlap.
            repeat_found |= len(set([element[0] for element in sample])) != len(sample)
            
            for i, element in enumerate(sample):
                if element[0] in boosted_priorities:
                    self.assertAlmostEqual(element[1],
                                           (higher_priority_numerator / (lower_priority_numerator + higher_priority_numerator)) * 1 / len(boosted_priorities),
                                           delta=self._delta)
                else:
                    self.assertAlmostEqual(element[1],
                                           (1 / (lower_priority_numerator + higher_priority_numerator)),
                                           delta=self._delta)
                    
                counts[element[0]] += 1
                sample_index_sets[i].add(element[0])

        if stratified:
            self.assertFalse(repeat_found)
            # Verify that all sample_index_sets are disjoint.
            for i in range(len(sample_index_sets)):
                for j in range(i):
                    self.assertEqual(len(sample_index_sets[i].intersection(sample_index_sets[j])), 0)
        else:
            self.assertTrue(repeat_found)
            # Verify that intersections exist among sample_index_sets.
            for i in range(len(sample_index_sets)):
                for j in range(i):
                    self.assertGreater(len(sample_index_sets[i].intersection(sample_index_sets[j])), 0)


        # This is not a very rigorous check. All counts should be
        # within 10% of expected.
        lower_priority_count = (iterations * sample_count) * (1 / (lower_priority_numerator + higher_priority_numerator))
        higher_priority_count = (iterations * sample_count) * (higher_priority_numerator / (lower_priority_numerator + higher_priority_numerator)) * 1 / len(boosted_priorities) if len(boosted_priorities) else 0
        lower_priority_count_delta = lower_priority_count * .1
        higher_priority_count_delta = higher_priority_count * .1
        for i, count in enumerate(counts):
            if i in boosted_priorities:
                self.assertAlmostEqual(count, higher_priority_count, delta=higher_priority_count_delta)
            else:
                self.assertAlmostEqual(count, lower_priority_count, delta=lower_priority_count_delta)
            
if __name__ == '__main__':
    unittest.main()
