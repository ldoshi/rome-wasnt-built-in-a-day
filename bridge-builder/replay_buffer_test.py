import numpy as np
import unittest
from replay_buffer import SumTree
from replay_buffer import ReplayBuffer

#class TestSumTree(unittest.TestCase):
class TestSumTree:
    
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

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self._delta = 1e-6
        self._beta = 1
        self._capacity = 15
        
    # This test case also verifies that returned indices correspond to
    # the correct entries.
    def test_uniform_replay(self):
        capacity = 10
        # The alpha and beta value do not make a difference in the uniform case.
        alphas = [.5, 1]
        betas = [.5, 1]
        for alpha in alphas:
            for beta in betas:
                replay_buffer = ReplayBuffer(capacity, alpha)

                for i in range(capacity):
                    replay_buffer.add_new_experience(i,i,i,i,i)
                    indices, states, actions, next_states, rewards, is_dones, weights = replay_buffer.sample(i+1, beta)
                    combined_list = [indices, states, actions, next_states, rewards, is_dones]

                    for combined_list_member in combined_list:
                        self.assertEqual(len(combined_list_member), i+1)

                    for j in range(i+1):
                        for combined_list_member in combined_list:
                            self.assertEqual(combined_list_member[j], j, "{} vs {}".format(combined_list_member, j))

                        self.assertEqual(weights[j], 1)

    def _make_replay_buffer_with_unit_priorities(self, alpha=1):
        replay_buffer = ReplayBuffer(self._capacity, alpha)

        for i in range(7):
            replay_buffer.add_new_experience(i,i,i,i,i)

        return replay_buffer
    
    # The exceptions_map consists of indices that do not have the
    # default frequency or weight and provides overrides with a
    # mapping of index->[frequency, weight].
    def _verify_sample(self, count, indices, weights, default_weight, exceptions_map):
        self.assertEqual(len(indices), count)
        self.assertEqual(len(weights), count)

        for i in range(count):
            if indices[i] in exceptions_map:
                self.assertGreater(exceptions_map[indices[i]][0], 0)
                self.assertAlmostEqual(weights[i], exceptions_map[indices[i]][1], delta=self._delta)
                exceptions_map[indices[i]][0] -= 1
            else:
                self.assertAlmostEqual(weights[i], default_weight, delta=self._delta)
            
        # Verify that all exceptions have been accounted for.
        for i in range(count):
            if indices[i] in exceptions_map:
                self.assertEqual(exceptions_map[indices[i]][0], 0)
                        
    def test_sample_large_counts(self):
        replay_buffer = self._make_replay_buffer_with_unit_priorities()
        # Sample a count higher then the entry count but lower than the capacity.
        indices, _, _, _, _, _, weights = replay_buffer.sample(self._capacity - 1, self._beta)
        self._verify_sample(count=self._capacity - 1, indices=indices, weights=weights, default_weight=1, exceptions_map={})
                
        # Sample a count higher than the capacity.
        indices, _, _, _, _, _, weights = replay_buffer.sample(self._capacity + 1, self._beta)
        self._verify_sample(count=self._capacity + 1, indices=indices, weights=weights, default_weight=1, exceptions_map={})
        
    def test_prioritized_replay_simple(self):
        replay_buffer = self._make_replay_buffer_with_unit_priorities()
        indices, _, _, _, _, _, weights = replay_buffer.sample(7, self._beta)
        self._verify_sample(count=7, indices=indices, weights=weights, default_weight=1, exceptions_map={})

        # Show update_priorities will shift returned samples, as
        # indicated in the exceptions_map. There are now 7 elements
        # with a total weight of 10. 4 units of weight belong to index
        # 2. With stratified sampling for 5 elements, we expect index
        # 2 to be represented exactly twice since each sample spans a
        # weight of 2 (and the index 2 was chosen to align strata
        # conveniently for deterministic testing).
        replay_buffer.update_priorities([2],[4])

        indices, _, _, _, _, _, weights = replay_buffer.sample(5, self._beta)
        self._verify_sample(count=5, indices=indices, weights=weights, default_weight=1, exceptions_map={2: [2, .25]})

        # Show new experiences now bear a higher probability (and
        # hence appear with a higher frequency), as indicated in the
        # exceptions_map. The new sample also has 4 units of weight,
        # giving the tree a total of 14. We extract 7 samples this
        # time to ensure a similar output as the previous case.
        replay_buffer.add_new_experience(7,7,7,7,7)
        indices, _, _, _, _, _, weights = replay_buffer.sample(7, self._beta)
        self._verify_sample(count=7, indices=indices, weights=weights, default_weight=1, exceptions_map={2: [2, .25], 7: [2, .25]})

    # Show that increasing and decreasing beta corresponding decreases
    # and increases the weights of higher priority elements.
    def test_prioritized_replay_betas(self):
        replay_buffer = self._make_replay_buffer_with_unit_priorities()

        replay_buffer.update_priorities([2],[4])

        _, _, _, _, _, _, weights = replay_buffer.sample(5, self._beta)

        _, _, _, _, _, _, higher_weights = replay_buffer.sample(5, self._beta / 2.)
        np.testing.assert_array_less(weights[1:3], higher_weights[1:3])
        
        _, _, _, _, _, _, lower_weights = replay_buffer.sample(5, 2 * self._beta)
        np.testing.assert_array_less(lower_weights[1:3], weights[1:3])
        
    # Show that increasing (decreasing) alpha increases (decreases)
    # the relative priority (and weight).
    def test_prioritized_replay_alphas(self):
        replay_buffer = self._make_replay_buffer_with_unit_priorities(alpha=1)
        replay_buffer.update_priorities([2],[4])
        indices, _, _, _, _, _, weights = replay_buffer.sample(5, self._beta)
                
        # Instead of the weight of 4 for index 2 and a total weight of
        # 10, we now have a weight of 2 for index 2 and a total weight
        # of 8. We adjust accordingly to take 4 samples for a
        # deterministic result for index 2.
        replay_buffer_half = self._make_replay_buffer_with_unit_priorities(alpha=.5)
        replay_buffer_half.update_priorities([2],[4])
        indices_half, _, _, _, _, _, weights_half = replay_buffer_half.sample(4, self._beta)
        self._verify_sample(count=4, indices=indices_half, weights=weights_half, default_weight=1, exceptions_map={2: [1, .5]})
        # The weight for the same priority is larger when the alpha is
        # smaller.
        self.assertGreater(weights_half[1], weights[1])

        # Instead of the weight of 4 for index 2 and a total weight of
        # 10, we now have a weight of 16 for index 2 and a total
        # weight of 22. We adjust accordingly to take 11 samples for a
        # deterministic result for index 2.
        replay_buffer_2 = self._make_replay_buffer_with_unit_priorities(alpha=2)
        replay_buffer_2.update_priorities([2],[4])
        indices_2, _, _, _, _, _, weights_2 = replay_buffer_2.sample(11, self._beta)
        self._verify_sample(count=11, indices=indices_2, weights=weights_2, default_weight=1, exceptions_map={2: [8, 1/16]})
        # The weight for the same priority is lower when the alpha is
        # larger.
        self.assertLess(weights_2[1], weights[1])

    def test_prioritized_replay_negative_td_errors(self):
        replay_buffer = self._make_replay_buffer_with_unit_priorities()
        replay_buffer.update_priorities([2],[4])
        _, _, _, _, _, _, weights = replay_buffer.sample(5, self._beta)

        replay_buffer_negative = self._make_replay_buffer_with_unit_priorities()
        replay_buffer.update_priorities([2],[-4])
        _, _, _, _, _, _, weights_negative = replay_buffer.sample(5, self._beta)

        np.testing.assert_array_equal(weights, weights_negative)
            
if __name__ == '__main__':
    unittest.main()
