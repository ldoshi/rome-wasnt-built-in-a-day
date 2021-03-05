import numpy as np
import random
from bisect import bisect_right

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


  def _sample_collector(self, samples, target_values, target_values_left, target_values_right, target_index, current_depth):
      if target_values_left == target_values_right:
          return

      assert target_index < self._capacity
      
      if current_depth == self._tree_depth:
          # The value (target_values_right - target_values_left)
          # represents how many times we selected this element
          # (sampling with replacement).
          for _ in range(target_values_right - target_values_left):
              samples.append((target_index, self._tree[layer_base(self._tree_depth - 1) + target_index] / self._tree_total_value()))
          return

      target_index *= 2
      left_sum = self._tree[layer_base(current_depth) + target_index]
      split_point = bisect_right(target_values, left_sum, target_values_left, target_values_right)
      # Traverse left.
      self._sample_collector(samples, target_values, target_values_left, split_point, target_index, current_depth + 1)

      # Traverse right.
      target_values[split_point:target_values_right] -= left_sum
      self._sample_collector(samples, target_values, split_point, target_values_right, target_index + 1, current_depth + 1)
          
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
      samples = []

      self._sample_collector(samples, target_values, target_values_left=0, target_values_right=len(target_values), target_index=0, current_depth=0)
      
      return samples
      
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
        
class ReplayBuffer:
    def __init__(self, capacity, alpha=.5):
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
    
