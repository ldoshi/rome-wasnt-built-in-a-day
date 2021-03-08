import numpy as np
import random
from bisect import bisect_right
from collections import deque

# The sum tree is contained in a single array. The slots represent the
# layers of the binary tree starting from the root laid out
# contiguously. The intermediate layers contains the sum of priorities
# of its children. The leaf nodes contain the priorities stored in the
# tree.
# An example tree with capacity 7. 
# [10,  6, 4,  2, 4, 2, 2,  1, 1, 3, 1, 1, 1, 2, _ ]
# |L0 |  L1  |     L2     |          L3          ^ unused slot.

def layer_base(depth):
    return (1 << depth) - 1

class SumTree:
  def __init__(self, capacity):      
      self._tree_depth = np.ceil(np.log2(capacity)).astype(int) + 1
      self._tree = np.zeros(1 << self._tree_depth)
      self._capacity = capacity
      self.max_value = 1

  # TODO(lyric): Delete this.
  def debug(self):
      print(self._tree_depth)
      print(self._tree)
      print(self._capacity)
      print(self.max_value)
      
  # By construction, the root node contains the sum of all values.
  def _tree_total_value(self):
      return self._tree[0]


  def _sample_collector(self, target_values):
      samples = []
      
      # Each tuple in traversal contains (target_values_left,
      # target_values_right, target_index, current_depth).
      traversal = deque([(0, len(target_values), 0, 0)])

      while traversal:
          current = traversal.pop()

          assert current[2] < self._capacity, "Target Index {} vs Capacity {}".format(current[2], self._capacity)
      
          if current[3] == self._tree_depth:
              # The value (target_values_right - target_values_left)
              # represents how many times we selected this element
              # (sampling with replacement).
              for _ in range(current[1] - current[0]):
                  samples.append((current[2], self._tree[layer_base(self._tree_depth - 1) + current[2]] / self._tree_total_value()))
              continue

          updated_target_index = current[2] * 2
          left_sum = self._tree[layer_base(current[3]) + updated_target_index]
          split_point = bisect_right(target_values, left_sum, current[0], current[1])
          # Traverse right.
          if split_point < current[1]:
              target_values[split_point:current[1]] -= left_sum
              traversal.append((split_point, current[1], updated_target_index + 1, current[3] + 1))

          # Traverse left.
          if current[0] < split_point:
              traversal.append((current[0], split_point, updated_target_index, current[3] + 1))

      return samples
          
  # Samples with replacement. Returns the index and probability of n
  # elements. If stratified is true, the selection space [0,
  # tree_total_value) is split into n contiguous buckets of size 1/n
  # and a sample if uniformly chosen from within each bucket.
  def sample(self, n, stratified=False):
      if n == 0:
          return []

      if stratified:
          boundaries = np.linspace(0,1,n+1) * self._tree_total_value()
          # Sorted by construction.
          target_values = [random.uniform(boundaries[i],boundaries[i+1]) for i in range(len(boundaries) - 1)]
      else:
          target_values = [random.uniform(0,1) * self._tree_total_value() for _ in range(n)]
          target_values.sort()

      target_values = np.array(target_values)
      return self._sample_collector(target_values)
            
  def set_value(self, index, value):
      if index >= self._capacity:
          raise ValueError('Attempted to set index {} beyond capacity {}.'.
                           format(index, self._capacity))

      self.max_value = max(value, self.max_value)
      

      current_depth_base = self._tree_depth - 1
      current_index = index

      delta = value - self._tree[layer_base(current_depth_base) + current_index]

      # Update nodes from leaf to root to adjust for the delta in the
      # value at index.
      while current_depth_base > -1:
          self._tree[layer_base(current_depth_base) + current_index] += delta

          current_depth_base -= 1
          current_index //= 2

# If update_priorities is never called, this calls implements uniform
# sampling of the Experience Replay Buffer.
class ReplayBuffer:
    def __init__(self, capacity, alpha=.5):
        self._capacity = capacity
        self._content = []
        self._alpha = alpha
        self._index = 0
        self._tree = SumTree(capacity)
        # epsilon is added to |td_errr| to ensure all priorities are non-zero.
        self._epsilon = 1e-10
  
    def add_new_experience(self, s, a, ss, r, d):
        if len(self._content) == self._capacity:    
            self._content[self._index] = [s, a, ss, r, d]
        else:                      
            self._content.append([s, a, ss, r, d])

        self._tree.set_value(self._index, self._tree.max_value)
        self._index = (self._index + 1) % self._capacity

    def update_priorities(self, indices, td_errors):
        for index, value in zip(indices, td_errors):
            self._tree.set_value(index, pow(abs(value) + self._epsilon, self._alpha))

    def sample(self, n, beta):
        samples = self._tree.sample(n, stratified=True)

        indices = []
        states = []
        actions = []
        next_states = []
        rewards = []
        is_dones = []
        weights = []
        # Collect samples and compute importance sampling weights.
        data_holders = [states, actions, next_states, rewards, is_dones]
        
        for index, probability in samples:
            indices.append(index)
            
            entry = self._content[index]
            for data, holder in zip(entry, data_holders):
                holder.append(data)

            weights.append(pow(1/len(self._content) * 1/probability, beta))

        weights = np.array(weights)
        return indices, np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(is_dones), (weights / np.max(weights))
