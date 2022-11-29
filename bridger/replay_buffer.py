import numpy as np
import torch

from bisect import bisect_right
from itertools import chain, starmap
from functools import partial
from collections import Counter

class SumTree:
    """A binary tree in which leaf node must contain a nonnegative value and
    each non-leaf node contains the sum of it's two children.

    Implementation Details: The sum tree is contained in a single array. The
    slots represent the layers of the binary tree starting from the root laid
    out contiguously. An example tree with capacity 7:

    [10,  6, 4,  2, 4, 2, 2,  1, 1, 3, 1, 1, 1, 2, _ ]
    |L0 |  L1  |     L2     |          L3          ^ unused slot.
    """

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

    # By construction, the root node contains the sum of all values.
    @property
    def _tree_total_value(self):
        return self._tree[0]

    def _process_intermediate_level(
        self,
        target_values,
        target_values_left,
        target_values_right,
        target_index,
        current_left_sum,
    ):
        """Helper function that partitions a list of cumulative probabilties
        between two children of a non-leaf node in the SumTree.

        Args:
            target_values: a numpy array of values in non-decreasing order to be
                used as cumulative probabilities for sampling from the SumTree
            target_values_left, target_values_right: two indices such that
                `target_values[target_values_left:target_values_right]` are the
                cumulative probabilities to be partitioned
            target_index: the index of the node within the SumTree, ordered
                canonically (left to right, top to bottom), between whose
                children the cumulative probabilities should be partitioned
            current_left_sum: the sum of all nodes to the left of `target_index`
                on the same level of the SumTree

        Returns up to two 4-tuples (one for each child node of `target_index`).
            Each 4-tuple will have the structure `(left, right, idx, lsum)`,
            where `idx` is the index of the child node within the SumTree,
            `lsum` is the sum of all nodes to the left of the child node on the
            same level, and `left` and `right` are indices such that [`left`, `right`)
            is a subset of [`target_values_left`, `target_values_right`) and
            `target_values[left:right]` are partitioned to the `idx` child node
        """

        assert (
            target_index < self._leaf_base
        ), f"Node {target_index} has no children and cannot be expanded"

        target_index = 2 * target_index + 1
        left_sum = self._tree[target_index] + current_left_sum
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

    def _sample_final_level(
        self,
        target_values,
        target_values_left,
        target_values_right,
        target_index,
        current_left_sum,
    ):
        """Helper function that takes a list of cumulative probabilties
        corresponding to a leaf node in the SumTree and returns a list
        containing an equal number of copies of the leaf node's index
        (among the leaf nodes only) and the probability corresponding to
        that leaf node.

        Args:
            target_values: a numpy array of values in non-decreasing order to be
                used as cumulative probabilities for sampling from the SumTree
            target_values_left, target_values_right: two indices such that
                `target_values[target_values_left:target_values_right]` are the
                cumulative probabilities corresponding to the leaf node
            target_index: the index of the leaf node within the SumTree,
                ordered canonically (left to right, top to bottom)
            current_left_sum: the sum of all leaf nodes to the left of this one

        Returns a list containing `(target_values_right - target_values_left)`
            copies of the tuple `(idx, p)` where `idx` is the index of this leaf
            node amongst all leaf nodes, and `p` is the probability of sampling
            this leaf node, as stored in the SumTree
        """

        leaf_index = target_index - self._leaf_base
        assert leaf_index >= 0, f"Node {target_index} is not a leaf"
        # The value (target_values_right - target_values_left) represents how
        # many times we selected this element (sampling with replacement).
        sample = (leaf_index, self._tree[target_index] / self._tree_total_value)
        return [sample] * (target_values_right - target_values_left)

    def _sample_collector(self, target_values):
        """Helper function that a takes a list of cumulative probabilties and
        returns the corresponding leaf node samples from the SumTree.

        Args:
            target_values: a numpy array of values in non-decreasing order to be
                used as cumulative probabilities for sampling from the SumTree

        Returns a list of samples `(idx, p)` from the SumTree, where `idx` is
            the leaf-level index of the leaf node sampled and `p` is the
            probability of sampling this leaf node, as stored in the SumTree.
        """

        level_func = partial(self._process_intermediate_level, target_values)
        sample_func = partial(self._sample_final_level, target_values)
        # Each tuple in splits contains (target_values_left,
        # target_values_right, target_index, current_left_sum).
        splits = [(0, len(target_values), 0, 0)]
        for i in range(self._tree_depth - 1):
            # With multiple cores, you can split out itertools.starmap
            # for a multiprocessing version
            splits = chain.from_iterable(starmap(level_func, splits))

        samples = chain.from_iterable(starmap(sample_func, splits))
        return list(samples)

    def sample(self, n, stratified=False):
        """Returns n samples drawn (with replacement) from the SumTree.

        Args:
            n: the number of samples to be drawn
            stratified: a boolean - if True, the range [0, 1) is split into `n`
                identical, contiguous buckets and a probability is drawn
                uniformly at random from each. if False, `n` probabilities are
                drawn uniformly at random from [0, 1)

        Returns a list of samples `(idx, p)` from the SumTree, where `idx` is
            the leaf-level index of the leaf node sampled and `p` is the
            probability of sampling this leaf node, as stored in the SumTree.
        """

        if n == 0:
            return []

        random_values = torch.rand(n).numpy()
        if stratified:
            # Sorted by construction.
            target_values = (random_values + np.arange(n)) / n
        else:
            target_values = np.sort(random_values)

        return self._sample_collector(target_values * self._tree_total_value)

    def set_value(self, index, value):
        """Sets the leaf node at leaf-level index `index` to have probability
        `value`, and updates the rest of the SumTree accordingly"""

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


# TODO: Code right now is clean and readable, but experimentation can be made
# more natural by splitting into separate but interdependent (custom) Sampler
# and Dataset classes
class ReplayBuffer(torch.utils.data.IterableDataset):
    """
    Attributes: 
        state_histogram: A dictionary mapping the number of state ids in the replay buffer to their counts.
    """
    def __init__(self, capacity: int, alpha: float=0.5, beta: float=0.5, batch_size: int=100):
        self._capacity = capacity
        self._alpha = alpha
        self._content = []
        self._index = 0
        self._tree = SumTree(capacity)
        # epsilon is added to |td_err| to ensure all priorities are non-zero.
        self._epsilon = 1e-10

        self.beta = beta
        self.batch_size = batch_size
        # A counter of the id counts currently in the buffer.
        self.state_histogram: Counter[int, int] = Counter()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        assert value > 0, "ReplayBuffer beta must be positive"
        self._beta = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        assert value > 0, "ReplayBuffer batch size must be positive"
        assert isinstance(value, int), "ReplayBuffer batch size must be an int"
        self._batch_size = value

    def __iter__(self):
        while True:
            samples = self._tree.sample(self.batch_size, stratified=True)
            prob_min = min(probability for index, probability in samples)
            for index, probability in samples:
                weight = pow(prob_min / probability, self.beta)
                # Currently we reserve the last value for the id of the state for logging purposes. If we add more values, consider adding a dataclass object when logging.
                yield (index, *self._content[index][:-1], weight)

    def add_new_experience(self, start_state, action, end_state, reward, success, state_id: int):
        experience = [start_state, action, end_state, reward, success, state_id]
        if len(self._content) == self._capacity:
            # Remove the previous state id from the Counter when overwriting
            self.state_histogram[self._content[self._index][-1]] -= 1
            self._content[self._index] = experience
        else:
            self._content.append(experience)

        self.state_histogram[state_id] += 1
        self._tree.set_value(self._index, self._tree.max_value)
        self._index = (self._index + 1) % self._capacity

    def update_priorities(self, indices, td_errors):
        # If update_priorities is never called, this call implements uniform
        # sampling of the Experience Replay Buffer.
        indices, td_errors = [
            (L if isinstance(L, list) else L.tolist()) for L in [indices, td_errors]
        ]
        for index, value in zip(indices, td_errors):
            self._tree.set_value(index, pow(abs(value) + self._epsilon, self._alpha))
