"""Tests replay buffer initializers."""
import unittest

from typing import Any

import numpy as np
from bridger import builder_trainer
from bridger import replay_buffer_initializers


_ENV_NAME = "gym_bridges.envs:Bridges-v0"
_ENV_WIDTH = 6
_REPLAY_BUFFER_CAPACITY = 10000


class ReplayBufferInitializersTest(unittest.TestCase):
    def setUp(self):
        self._replay_buffer = []
        self._env = builder_trainer.make_env(
            name=_ENV_NAME, width=_ENV_WIDTH, force_standard_config=True
        )

    def add_new_experience(
        self, start_state: Any, action: Any, end_state: Any, reward: Any, success: Any
    ):
        self._replay_buffer.append([start_state, action, end_state, reward, success])

    def test_only_reset_state(self):
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_ONLY_RESET_STATE,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )

        self.assertEqual(len(self._replay_buffer), _ENV_WIDTH)
        reset_state = self._env.reset()
        for experience, expected_action in zip(self._replay_buffer, range(_ENV_WIDTH)):
            np.testing.assert_array_equal(experience[0], reset_state)
            self.assertEqual(experience[1], expected_action)

    def test_standard_configuration_bridge_states(self):
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_STANDARD_CONFIGURATION_BRIDGE_STATES,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )
        self.assertEqual(
            len(self._replay_buffer),
            _ENV_WIDTH * ((int((_ENV_WIDTH - 2) / 2) + 1) ** 2 - 1),
        )
        count_dones = 0
        for experience in self._replay_buffer:
            if experience[4]:
                count_dones += 1
        self.assertEqual(count_dones, 2)

    def test_2_bricks(self):
        """Verifies the n-bricks strategy with 2 bricks.

        We chose 2 so we can reason about all the experiences.
        """
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )
        # The first brick produces one of 6 unique experiences:
        # (initial state, <each action>). We now have 3 distinct
        # starting points: (a) brick on the left, (b) brick on the
        # right, and (c) what looks like the initial state. (a) and
        # (b) will produce an additional 6 unique experiences
        # each. (c) will not produce any unique experiences.
        self.assertEqual(len(self._replay_buffer), 18)
        count_dones = 0
        for experience in self._replay_buffer:
            if experience[4]:
                count_dones += 1
        self.assertEqual(count_dones, 0)

    def test_4_bricks(self):
        """Verifies the n-bricks strategy with 4 bricks.

        We chose 4 so we can reason about the number of dones.
        """
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_4_BRICKS,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )
        # This number is empirically derived.
        self.assertEqual(len(self._replay_buffer), 132)
        count_dones = 0
        for experience in self._replay_buffer:
            if experience[4]:
                count_dones += 1

        # The 2 done scenarios occur when the last brick is placed
        # with action 1 after {0, [4,3]} or action 3 after {[0,1],
        # 4}. The actions in [] must happen in that order relative to
        # other list members while {} means the order doesn't matter.
        self.assertEqual(count_dones, 2)

    def test_capacity_too_small(self):
        self.assertRaisesRegex(
            ValueError,
            "initialized with more experiences than capacity",
            replay_buffer_initializers.initialize_replay_buffer,
            strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
            replay_buffer_capacity=2,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )

    def test_illegal_strategy(self):
        self.assertRaisesRegex(
            ValueError,
            "Unrecognized replay buffer initialization strategy",
            replay_buffer_initializers.initialize_replay_buffer,
            strategy="illegal strategy",
            replay_buffer_capacity=2,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )


if __name__ == "__main__":
    unittest.main()