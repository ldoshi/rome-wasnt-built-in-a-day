"""Tests replay buffer initializers."""
import unittest

from typing import Any, NamedTuple

import numpy as np
import pathlib
import shutil
import torch
import os

from bridger import (
    builder_trainer,
    hash_utils,
    replay_buffer,
    replay_buffer_initializers,
)
from bridger.logging_utils import object_logging


_ENV_NAME = "gym_bridges.envs:Bridges-v0"
_ENV_WIDTH = 6
_REPLAY_BUFFER_CAPACITY = 10000

_TMP_DIR = "tmp/nested_tmp"
_LOG_FILENAME = "log_filename"


class Experience(NamedTuple):
    start_state: Any
    action: Any
    end_state: Any
    reward: Any
    success: Any
    state_id: int


def create_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    path.mkdir(parents=True, exist_ok=True)


def delete_temp_dir():
    path = pathlib.Path(_TMP_DIR)
    shutil.rmtree(path.parts[0], ignore_errors=True)


class ReplayBufferInitializersTest(unittest.TestCase):
    def setUp(self):
        create_temp_dir()

    def tearDown(self):
        delete_temp_dir()

    def setUp(self):
        self._replay_buffer = []
        self._env = builder_trainer.make_env(
            name=_ENV_NAME, width=_ENV_WIDTH, force_standard_config=True
        )

    def add_new_experience(
        self,
        start_state: Any,
        action: Any,
        end_state: Any,
        reward: Any,
        success: Any,
        state_id: int,
    ):
        self._replay_buffer.append(
            Experience(
                start_state=start_state,
                action=action,
                end_state=end_state,
                reward=reward,
                success=success,
                state_id=state_id,
            )
        )

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
            np.testing.assert_array_equal(experience.start_state, reset_state)
            self.assertEqual(experience.action, expected_action)

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
        count_successes = 0
        for experience in self._replay_buffer:
            if experience.success:
                count_successes += 1
        self.assertEqual(count_successes, 2)

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
        count_successes = 0
        for experience in self._replay_buffer:
            if experience.success:
                count_successes += 1
        self.assertEqual(count_successes, 0)

    def test_4_bricks(self):
        """Verifies the n-bricks strategy with 4 bricks.

        We chose 4 so we can reason about the number of successes.
        """
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_4_BRICKS,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=self.add_new_experience,
        )
        # This number is empirically derived.
        self.assertEqual(len(self._replay_buffer), 132)
        count_successes = 0
        for experience in self._replay_buffer:
            if experience.success:
                count_successes += 1

        # The 2 success scenarios occur when the last brick is placed
        # with action 1 after {0, [4,3]} or action 3 after {[0,1],
        # 4}. The actions in [] must happen in that order relative to
        # other list members while {} means the order doesn't matter.
        self.assertEqual(count_successes, 2)

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

    def test_real_replay_buffer(self):
        buffer = replay_buffer.ReplayBuffer(
            capacity=_REPLAY_BUFFER_CAPACITY,
        )
        replay_buffer_initializers.initialize_replay_buffer(
            strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=buffer.add_new_experience,
        )
        self.assertIsNone(buffer.state_histogram)

    def test_real_replay_buffer_extra_logger(self):
        buffer = replay_buffer.ReplayBuffer(
            capacity=_REPLAY_BUFFER_CAPACITY,
        )

        with object_logging.ObjectLogManager(
            object_logging_base_dir=os.path.dirname(_TMP_DIR),
            experiment_name=os.path.basename(_TMP_DIR),
        ) as logger:
            state_logger = object_logging.LoggerAndNormalizer(
                log_filename=_LOG_FILENAME,
                object_log_manager=logger,
                log_entry_object_class=torch.Tensor,
                make_hashable_fn=hash_utils.hash_tensor,
            )

            self.assertRaisesRegex(
                AssertionError,
                "state id should not be passed if the replay buffer",
                replay_buffer_initializers.initialize_replay_buffer,
                strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
                replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
                env=self._env,
                add_new_experience=buffer.add_new_experience,
                state_logger=state_logger,
            )

    def test_real_replay_buffer_debug(self):
        buffer = replay_buffer.ReplayBuffer(
            capacity=_REPLAY_BUFFER_CAPACITY,
            debug=True,
        )

        with object_logging.ObjectLogManager(
            object_logging_base_dir=os.path.dirname(_TMP_DIR),
            experiment_name=os.path.basename(_TMP_DIR),
        ) as logger:
            state_logger = object_logging.LoggerAndNormalizer(
                log_filename=_LOG_FILENAME,
                object_log_manager=logger,
                log_entry_object_class=torch.Tensor,
                make_hashable_fn=hash_utils.hash_tensor,
            )

            replay_buffer_initializers.initialize_replay_buffer(
                strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
                replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
                env=self._env,
                add_new_experience=buffer.add_new_experience,
                state_logger=state_logger,
            )

        # As mentioned in test_2_bricks, there are 3 distinct starting
        # states.
        self.assertEqual(len(buffer.state_histogram), 3)

    def test_real_replay_buffer_debug_no_state_id(self):
        buffer = replay_buffer.ReplayBuffer(
            capacity=_REPLAY_BUFFER_CAPACITY,
            debug=True,
        )
        self.assertRaisesRegex(
            AssertionError,
            "state id must be passed",
            replay_buffer_initializers.initialize_replay_buffer,
            strategy=replay_buffer_initializers.STRATEGY_2_BRICKS,
            replay_buffer_capacity=_REPLAY_BUFFER_CAPACITY,
            env=self._env,
            add_new_experience=buffer.add_new_experience,
        )


if __name__ == "__main__":
    unittest.main()
