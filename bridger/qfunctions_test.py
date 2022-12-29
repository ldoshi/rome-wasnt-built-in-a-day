"""Tests for Q-learning modules"""
import unittest

from typing import NamedTuple

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

_IMAGE_HEIGHT =3
_IMAGE_WIDTH =3 
_NUM_ACTIONS = 4

class QAndTargetValues(NamedTuple):
    q_value_0: torch.Tensor
    target_value_0: torch.Tensor
    q_value_1: torch.Tensor
    target_value_1: torch.Tensor
    q_value_2: torch.Tensor
    target_value_2: torch.Tensor

class CNNQManagerTest(unittest.TestCase):


    def _get_q_and_target_values(self,q_manager: qfunctions.CNNQManager) -> QAndTargetValues:
        x = torch.ones(_IMAGE_HEIGHT, _IMAGE_WIDTH)

        q_value_0 = q_manager.q(x)
        target_value_0 = q_manager.target(x)

        # The q and target evaluate the same to start.
        self.assertTrue(torch.all(q_value_0 == target_value_0))

        q_params = q_manager.q.state_dict()
        q_params[next(iter(q_params))] *= 2
        q_manager.q.load_state_dict(q_params)

        q_value_1 = q_manager.q(x)
        target_value_1 = q_manager.target(x)

        q_manager.update_target()
        q_value_2 = q_manager.q(x)
        target_value_2 = q_manager.target(x)

        # Calling update_target never affects q itself.
        self.assertTrue(torch.all(q_value_1 == q_value_2))

        return QAndTargetValues(q_value_0=q_value_0, target_value_0=target_value_0,
                                q_value_1=q_value_1, target_value_1=target_value_1,
                                q_value_2=q_value_2, target_value_2=target_value_2)
    
    def test_tau_0(self):
        q_manager = qfunctions.CNNQManager(image_height=_IMAGE_HEIGHT, image_width=_IMAGE_WIDTH, num_actions=_NUM_ACTIONS, tau=0)
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tao as 0, the target is unchanged after update_target.
        self.assertTrue(torch.all(values.target_value_1 == values.target_value_2))

    def test_tau_1(self):
        q_manager = qfunctions.CNNQManager(image_height=_IMAGE_HEIGHT, image_width=_IMAGE_WIDTH, num_actions=_NUM_ACTIONS, tau=1)
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tao as 1, the target now matches q after update_target.
        self.assertTrue(torch.all(values.q_value_2 == values.target_value_2))
                
    def test_tau_intermediate(self):
        q_manager = qfunctions.CNNQManager(image_height=_IMAGE_HEIGHT, image_width=_IMAGE_WIDTH, num_actions=_NUM_ACTIONS, tau=.7)
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tao as .7, the target has changed and also does not
        # match q after update_target.
        self.assertFalse(torch.all(values.target_value_1 == values.target_value_2))
        self.assertFalse(torch.all(values.q_value_2 == values.target_value_2))


    def test_tau_none(self):
        q_manager = qfunctions.CNNQManager(image_height=_IMAGE_HEIGHT, image_width=_IMAGE_WIDTH, num_actions=_NUM_ACTIONS, tau=None)
        values = self._get_q_and_target_values(q_manager)

        # The q value changes and target evaluates the same network.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertFalse(torch.all(values.target_value_0 == values.target_value_1))
        self.assertTrue(torch.all(values.q_value_1 == values.target_value_1))

        # With tao as None, update_target does not do anything.
        self.assertTrue(torch.all(values.target_value_1 == values.target_value_2))
        self.assertTrue(torch.all(values.q_value_2 == values.target_value_2))

        
        
if __name__ == "__main__":
    unittest.main()

