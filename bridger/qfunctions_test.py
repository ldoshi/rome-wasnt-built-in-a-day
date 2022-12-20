"""Tests for Q-learning modules"""
import unittest

import torch
import bridger.qfunctions as qfunctions


class QFunctionsTest(unittest.TestCase):
    """Validate helper functions for Q-learning modules"""

    def test_encode_one_hot_1x2x3(self):
        """Validate encoding 3-dim index tensor to one-hot"""
        index_tensor = torch.tensor([[[0, 1, 2], [1, 0, 1]]])
        expected_tensor = torch.tensor(
            [[[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [1, 0, 1]], [[0, 0, 1], [0, 0, 0]]]]
        )
        one_hot = qfunctions.encode_enum_state_to_channels(index_tensor, 3)
        self.assertTrue(torch.equal(one_hot, expected_tensor))

    def test_encode_one_hot_2x1x2(self):
        """Validate encoding 3-dim index tensor (with 2 batches) to one-hot"""
        index_tensor = torch.tensor([[[0, 2]], [[2, 1]]])
        expected_tensor = torch.tensor(
            [[[[1, 0]], [[0, 0]], [[0, 1]]], [[[0, 0]], [[0, 1]], [[1, 0]]]]
        )
        one_hot = qfunctions.encode_enum_state_to_channels(index_tensor, 3)
        self.assertTrue(torch.equal(one_hot, expected_tensor))


if __name__ == "__main__":
    unittest.main()
