import numpy as np
import random
from bisect import bisect_right
from itertools import chain, starmap
from functools import partial

# The sum tree is contained in a single array. The slots represent the
# layers of the binary tree starting from the root laid out
# contiguously. The intermediate layers contains the sum of priorities
# of its children. The leaf nodes contain the priorities stored in the
# tree.
# An example tree with capacity 7.
# [10,  6, 4,  2, 4, 2, 2,  1, 1, 3, 1, 1, 1, 2, _ ]
# |L0 |  L1  |     L2     |          L3          ^ unused slot.


class SumTree:
    def __init__(self, capacity):
        """Creates a SumTree for storing up to `capacity` leaf values"""
        self._tree_depth = np.ceil(np.log2(capacity)).astype(int) + 1
        self._leaf_base = (1 << (self._tree_depth - 1)) - 1
        self._tree = np.zeros(2 * self._leaf_base + 1)
        self._capacity = capacity
        self.max_value = 1

    # TODO(lyric): Delete this.
    def debug(self):
        print(self._tree_depth)
        print(self._tree)
        print(self._capacity)
        print(self.max_value)

    def __getitem__(self, index):
        return self._tree[index]

    # By construction, the root node contains the sum of all values.
    @property
    def _tree_total_value(self):
        return self[0]

    def _process_layer(
        self,
        target_values,
        target_values_left,
        target_values_right,
        target_index,
        current_left_sum,
    ):

        cap_err = f"Target Index {target_index} vs Capacity {self._capacity}"

        leaf_index = target_index - self._leaf_base
        assert leaf_index < self._capacity, cap_err

        if leaf_index >= 0:
            # The value (target_values_right - target_values_left)
            # represents how many times we selected this element
            # (sampling with replacement).
            sample = (leaf_index, self[target_index] / self._tree_total_value)
            return [sample] * (target_values_right - target_values_left)

        target_index = 2 * target_index + 1
        left_sum = self[target_index] + current_left_sum
        split_point = bisect_right(
            target_values, left_sum, target_values_left, target_values_right
        )

        next_level_splits = []
        # Traverse left.
        if target_values_left < split_point:
            next_level_splits.append(
                (target_values_left, split_point, target_index, current_left_sum)
            )

        # Traverse right.
        if split_point < target_values_right:
            next_level_splits.append(
                (split_point, target_values_right, target_index + 1, left_sum)
            )

        return next_level_splits

    def _sample_collector(self, target_values):

        # Each tuple in splits contains (target_values_left,
        # target_values_right, target_index, current_left_sum).
        layer_func = partial(self._process_layer, target_values)
        splits = [(0, len(target_values), 0, 0)]
        for i in range(self._tree_depth):
            # With multiple cores, you can split out itertools.starmap
            # for a multiprocessing version
            splits = chain.from_iterable(starmap(layer_func, splits))
        return list(splits)

    # Samples with replacement. Returns the index and probability of n
    # elements. If stratified is true, the selection space [0,
    # tree_total_value) is split into n contiguous buckets of size 1/n
    # and a sample if uniformly chosen from within each bucket.
    def sample(self, n, stratified=False):
        if n == 0:
            return []

        if stratified:
            # Sorted by construction.
            target_values = (np.random.rand(n) + np.arange(n)) / n
        else:
            target_values = np.sort(np.random.rand(n))

        return self._sample_collector(target_values * self._tree_total_value)

    def set_value(self, index, value):
        if index >= self._capacity:
            raise ValueError(
                f"Attempted to set index {index} beyond capacity {self._capacity}."
            )

        self.max_value = max(value, self.max_value)

        index += self._leaf_base
        delta = value - self._tree[index]

        # Update nodes from leaf to root to adjust for the delta in the
        # value at index.
        while index >= 0:
            self._tree[index] += delta
            index = (index - 1) // 2


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.5):
        self._size = n
        self._content = []
        self._index = 0

    def add(self, s, a, ss, r, d):
        if len(self._content) == self._size:
            self._content[self._index] = [s, a, ss, r, d]
            self._index = (self._index + 1) % self._size
        else:
            self._content.append([s, a, ss, r, d])

    def sample(self, n, beta):
        samples = random.sample(self._content, min(n, len(self._content)))

        states = np.array([i[0] for i in samples])
        actions = np.array([i[1] for i in samples])
        next_states = np.array([i[2] for i in samples])
        rewards = np.array([[i[3]] for i in samples])
        is_dones = np.array([[i[4]] for i in samples])
        return states, actions, next_states, rewards, is_dones
