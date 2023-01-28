"""Tests for Q-learning modules"""

import unittest

from typing import NamedTuple

from parameterized import parameterized

import torch
from bridger import builder_trainer, hash_utils, qfunctions


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


_ENV_WIDTH = 4
_NUM_ACTIONS = 4
_ENV_NAME = "gym_bridges.envs:Bridges-v0"
_ENV = builder_trainer.make_env(
    name=_ENV_NAME, width=_ENV_WIDTH, force_standard_config=True
)
_BRICK_COUNT = 3


class QAndTargetValues(NamedTuple):
    q_value_0: torch.Tensor
    target_value_0: torch.Tensor
    q_value_1: torch.Tensor
    target_value_1: torch.Tensor
    q_value_2: torch.Tensor
    target_value_2: torch.Tensor


class QManagerTest(unittest.TestCase):
    def _get_q_and_target_values(
        self, q_manager: qfunctions.QManager
    ) -> QAndTargetValues:
        x = torch.Tensor(_ENV.reset())

        q_value_0 = q_manager.q(x)
        target_value_0 = q_manager.target(x)

        # The q and target evaluate the same to start.
        self.assertTrue(torch.all(q_value_0 == target_value_0))

        q_params = q_manager.q.state_dict()

        # We must access the entries for x explicitly in the tabular
        # case. Manipulating any weights is sufficient in the neural
        # network case. The x_hashed value relies on implementation
        # details for creating keys from both TabularQManager and
        # ParameterDict. This is suboptimal but was practical.
        x_hashed = f"_q.{str(hash_utils.hash_tensor(x.int()))}"
        if x_hashed in q_params:
            params_key = x_hashed
        else:
            params_key = next(iter(q_params))

        q_params[params_key] *= 2
        q_manager.q.load_state_dict(q_params)

        q_value_1 = q_manager.q(x)
        target_value_1 = q_manager.target(x)

        q_manager.update_target()
        q_value_2 = q_manager.q(x)
        target_value_2 = q_manager.target(x)

        # Calling update_target never affects q itself.
        self.assertTrue(torch.all(q_value_1 == q_value_2))

        return QAndTargetValues(
            q_value_0=q_value_0,
            target_value_0=target_value_0,
            q_value_1=q_value_1,
            target_value_1=target_value_1,
            q_value_2=q_value_2,
            target_value_2=target_value_2,
        )

    @parameterized.expand(
        [
            (
                "CNNQManager",
                qfunctions.CNNQManager(
                    image_height=_ENV.shape[0],
                    image_width=_ENV.shape[1],
                    num_actions=_NUM_ACTIONS,
                    tau=0,
                ),
            ),
            (
                "TabularQManager",
                qfunctions.TabularQManager(
                    env=_ENV,
                    brick_count=_BRICK_COUNT,
                    tau=0,
                ),
            ),
        ]
    )
    def test_tau_0(self, name: str, q_manager: qfunctions.QManager):
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tau as 0, the target is unchanged after update_target.
        self.assertTrue(torch.all(values.target_value_1 == values.target_value_2))

    @parameterized.expand(
        [
            (
                "CNNQManager",
                qfunctions.CNNQManager(
                    image_height=_ENV.shape[0],
                    image_width=_ENV.shape[1],
                    num_actions=_NUM_ACTIONS,
                    tau=1,
                ),
            ),
            (
                "TabularQManager",
                qfunctions.TabularQManager(
                    env=_ENV,
                    brick_count=_BRICK_COUNT,
                    tau=1,
                ),
            ),
        ]
    )
    def test_tau_1(self, name: str, q_manager: qfunctions.QManager):
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tau as 1, the target now matches q after update_target.
        self.assertTrue(torch.all(values.q_value_2 == values.target_value_2))

    @parameterized.expand(
        [
            (
                "CNNQManager",
                qfunctions.CNNQManager(
                    image_height=_ENV.shape[0],
                    image_width=_ENV.shape[1],
                    num_actions=_NUM_ACTIONS,
                    tau=0.7,
                ),
            ),
            (
                "TabularQManager",
                qfunctions.TabularQManager(
                    env=_ENV,
                    brick_count=_BRICK_COUNT,
                    tau=0.7,
                ),
            ),
        ]
    )
    def test_tau_intermediate(self, name: str, q_manager: qfunctions.QManager):
        values = self._get_q_and_target_values(q_manager)

        # The q value changes, but the target has not been updated yet.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertTrue(torch.all(values.target_value_0 == values.target_value_1))

        # With tau as .7, the target has changed and also does not
        # match q after update_target.
        self.assertFalse(torch.all(values.target_value_1 == values.target_value_2))
        self.assertFalse(torch.all(values.q_value_2 == values.target_value_2))

    @parameterized.expand(
        [
            (
                "CNNQManager",
                qfunctions.CNNQManager(
                    image_height=_ENV.shape[0],
                    image_width=_ENV.shape[1],
                    num_actions=_NUM_ACTIONS,
                    tau=None,
                ),
            ),
            (
                "TabularQManager",
                qfunctions.TabularQManager(
                    env=_ENV,
                    brick_count=_BRICK_COUNT,
                    tau=None,
                ),
            ),
        ]
    )
    def test_tau_none(self, name: str, q_manager: qfunctions.QManager):
        values = self._get_q_and_target_values(q_manager)

        # The q value changes and target evaluates the same network.
        self.assertFalse(torch.all(values.q_value_0 == values.q_value_1))
        self.assertFalse(torch.all(values.target_value_0 == values.target_value_1))
        self.assertTrue(torch.all(values.q_value_1 == values.target_value_1))

        # With tau as None, update_target does not do anything.
        self.assertTrue(torch.all(values.target_value_1 == values.target_value_2))
        self.assertTrue(torch.all(values.q_value_2 == values.target_value_2))

    @parameterized.expand(
        [
            (
                "CNNQ",
                qfunctions.CNNQ(
                    image_height=_ENV.shape[0],
                    image_width=_ENV.shape[1],
                    num_actions=_NUM_ACTIONS,
                ),
            ),
            (
                "TabularQ",
                qfunctions.TabularQ(
                    env=_ENV,
                    brick_count=_BRICK_COUNT,
                ),
            ),
        ]
    )
    def test_forward(self, name: str, q: torch.nn.Module):
        q = qfunctions.TabularQ(env=_ENV, brick_count=_BRICK_COUNT)

        x = torch.Tensor(_ENV.reset())

        value_0 = q(x)
        self.assertEqual(value_0.shape, (_ENV_WIDTH,))

        value_1 = q(x)
        self.assertTrue(torch.all(value_0 == value_1))
        self.assertIsNot(value_0, value_1)

        x_batch = x[None, :]

        batch_value_0 = q(x_batch)
        self.assertEqual(
            batch_value_0.shape,
            (
                1,
                _ENV_WIDTH,
            ),
        )

        batch_value_1 = q(x_batch)
        self.assertTrue(torch.all(batch_value_0 == batch_value_1))
        self.assertIsNot(batch_value_0, batch_value_1)
        self.assertIsNot(batch_value_0[0], batch_value_1[0])

        self.assertTrue(torch.all(batch_value_0[0] == value_0))
        self.assertIsNot(batch_value_0[0], batch_value_0)


if __name__ == "__main__":
    unittest.main()
