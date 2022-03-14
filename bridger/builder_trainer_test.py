"""Tests for core building and training components."""
import unittest

import pathlib
import itertools
import numpy as np
import shutil
from parameterized import parameterized
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping

import torch

from bridger import builder
from bridger import builder_trainer
from bridger import policies
from bridger.logging import object_logging
from bridger.logging import object_log_readers
from bridger.logging import log_entry


_ENV_NAME = "gym_bridges.envs:Bridges-v0"
_OBJECT_LOGGING_DIR = "tmp_object_logging_dir"
_DELTA = 1e-6


def _get_model(
    object_log_manager: object_logging.ObjectLogManager,
    debug: bool = False,
    max_episode_length=1,
    batch_size=5,
    initial_memories_count=1000,
) -> builder_trainer.BridgeBuilderModel:
    return builder_trainer.BridgeBuilderModel(
        object_log_manager,
        env_width=3,
        env_force_standard_config=True,
        seed=12345,
        max_episode_length=max_episode_length,
        val_batch_size=1,
        batch_size=batch_size,
        object_logging_dir=_OBJECT_LOGGING_DIR,
        initial_memories_count=initial_memories_count,
        debug=debug,
    )


def _get_trainer(max_steps: int = 1, callbacks: list[Callback] = None) -> Trainer:
    return Trainer(
        val_check_interval=1,
        # The validation batch size can be adjusted via a config, but
        # we only need a single batch.
        limit_val_batches=1,
        max_steps=max_steps,
        callbacks=callbacks,
    )


class BridgeBuilderTrainerTest(unittest.TestCase):
    """Verifies training hooks and structure."""

    def tearDown(self):
        # TODO: Make a more coherent plan for writing test output to a temp dir
        #       and retaining it on failure
        shutil.rmtree("lightning_logs", ignore_errors=True)
        shutil.rmtree(_OBJECT_LOGGING_DIR, ignore_errors=True)

    def test_validation_builder(self):
        """Ensures ValidationBuilder keeps building and returning results."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=4, force_standard_config=True
        )

        def _constant_estimator(state) -> torch.Tensor:
            return torch.tensor([1, 0, 0, 0])

        # Get an arbitrary number of results. The ValidationBuilder
        # will keep producing results until we stop asking.
        for build_result in itertools.islice(
            builder_trainer.ValidationBuilder(
                env=env,
                policy=policies.GreedyPolicy(_constant_estimator),
                episode_length=1,
            ),
            10,
        ):
            self.assertFalse(build_result[0])
            self.assertEqual(build_result[1], -1)
            self.assertEqual(build_result[2], 1)

    @parameterized.expand(
        [
            ("No Early Stopping", []),
            (
                "Early Stopping",
                [
                    EarlyStopping(
                        monitor="val_reward",
                        patience=1,
                        mode="max",
                        strict=True,
                        check_on_train_epoch_end=False,
                    )
                ],
            ),
        ]
    )
    def test_early_stopping(
        self,
        name: str,
        early_stopping_callback: list[Callback],
    ):
        """Checks early stopping callback actually stops training.

        The training should stop before max_steps if early stopping is
        triggered. The exact number of steps may vary as the network
        architecture evolves.
        """

        class CountingCallback(Callback):
            """Callback that counts the number of training batches."""

            def __init__(self):
                self.count = 0

            def on_validation_end(self, trainer, model):
                if not trainer.sanity_checking:
                    self.count += 1

        max_steps = 50
        callbacks = [CountingCallback()] + early_stopping_callback

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer(max_steps, callbacks).fit(_get_model(object_log_manager))

        if early_stopping_callback:
            self.assertLess(callbacks[0].count, max_steps)
        else:
            self.assertEqual(callbacks[0].count, max_steps)

    def test_train_no_logging_without_debug(self):
        """Verifies that training history and batches are not logged by default."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer().fit(_get_model(object_log_manager))
        path = pathlib.Path(_OBJECT_LOGGING_DIR)
        self.assertTrue(path.is_dir())
        self.assertFalse(list(path.iterdir()))

    def _verify_log_entries(self, expected_entries, logged_entries):
        """Asserts equality for expected_entries and logged_entries.

        Checks equality field-by-field for each pair of corresponding entries.
        """
        self.assertEqual(len(expected_entries), len(logged_entries))
        for expected_entry, logged_entry in zip(expected_entries, logged_entries):
            for field in expected_entry.__dataclass_fields__:
                expected_entry_value = getattr(expected_entry, field)
                logged_entry_value = getattr(logged_entry, field)
                if isinstance(expected_entry_value, torch.Tensor):
                    if field == "loss":
                        self.assertTrue(
                            torch.allclose(
                                expected_entry_value, logged_entry_value, atol=1e-4
                            )
                        )
                    else:
                        self.assertTrue(
                            torch.equal(expected_entry_value, logged_entry_value)
                        )
                    continue

                if isinstance(expected_entry_value, float):
                    self.assertAlmostEqual(
                        expected_entry_value, logged_entry_value, delta=_DELTA
                    )
                    continue

                self.assertEqual(expected_entry_value, logged_entry_value)

    def test_training_batch_logging(self):
        """Verifies that training batches are logged in debug mode."""

        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer().fit(
                _get_model(object_log_manager=object_log_manager, debug=True)
            )
        expected_entries = [
            log_entry.TrainingBatchLogEntry(
                batch_idx=0,
                indices=torch.tensor([127, 231, 516, 661, 863]),
                state_ids=[1, 1, 1, 1, 1],
                actions=torch.tensor([1, 0, 1, 1, 1]),
                next_state_ids=[2, 3, 2, 2, 2],
                rewards=torch.tensor([-1, -1, -1, -1, -1]),
                successes=torch.tensor([False, False, False, False, False]),
                weights=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64),
                loss=torch.tensor(0.9522, dtype=torch.float64, requires_grad=True),
            )
        ]

        logged_entries = list(
            object_log_readers.read_object_log(
                _OBJECT_LOGGING_DIR, log_entry.TRAINING_BATCH_LOG_ENTRY
            )
        )

        self._verify_log_entries(expected_entries, logged_entries)

    def test_training_history_td_error_logging(self):
        """Verifies that td errors are logged in debug mode."""

        max_steps = 2
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer(max_steps=max_steps).fit(
                _get_model(
                    object_log_manager=object_log_manager, debug=True, batch_size=2
                )
            )
        expected_entries = [
            log_entry.TrainingHistoryTDErrorLogEntry(
                batch_idx=0, state_id=1, action=1, td_error=-0.998901
            ),
            log_entry.TrainingHistoryTDErrorLogEntry(
                batch_idx=0, state_id=1, action=2, td_error=-4.946752
            ),
            log_entry.TrainingHistoryTDErrorLogEntry(
                batch_idx=1, state_id=1, action=0, td_error=-0.840642
            ),
            log_entry.TrainingHistoryTDErrorLogEntry(
                batch_idx=1, state_id=1, action=1, td_error=-0.924127
            ),
        ]

        logged_entries = list(
            object_log_readers.read_object_log(
                _OBJECT_LOGGING_DIR, log_entry.TRAINING_HISTORY_TD_ERROR_LOG_ENTRY
            )
        )

        self._verify_log_entries(expected_entries, logged_entries)

    def test_training_history_q_value_logging_exact(self):
        """Verifies that td errors are logged in debug mode.

        This test verifies the q and q target values exactly for a simple case."""

        max_steps = 1
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer(max_steps=max_steps).fit(
                _get_model(
                    object_log_manager=object_log_manager, debug=True, batch_size=2
                )
            )
        expected_entries = [
            log_entry.TrainingHistoryQValueLogEntry(
                batch_idx=0,
                state_id=0,
                action=0,
                q_value=-0.128640,
                q_target_value=-0.091397,
            ),
            log_entry.TrainingHistoryQValueLogEntry(
                batch_idx=0,
                state_id=0,
                action=1,
                q_value=-0.045778,
                q_target_value=0.028986,
            ),
            log_entry.TrainingHistoryQValueLogEntry(
                batch_idx=0,
                state_id=0,
                action=2,
                q_value=-0.165802,
                q_target_value=-0.025232,
            ),
        ]

        logged_entries = list(
            object_log_readers.read_object_log(
                _OBJECT_LOGGING_DIR, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY
            )
        )

        self._verify_log_entries(expected_entries, logged_entries)

    def test_training_history_q_value_logging_general(self):
        """Verifies that td errors are logged in debug mode.

        This test executes over multiple steps of a longer episodes,
        generating entries across more states. We verify that the
        expected pattern of (batch_id, state_id, action) triples
        appear in the log.

        """

        max_steps = 2
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer(max_steps=max_steps).fit(
                _get_model(
                    object_log_manager=object_log_manager,
                    debug=True,
                    max_episode_length=2,
                    batch_size=2,
                )
            )

        logged_entries = list(
            object_log_readers.read_object_log(
                _OBJECT_LOGGING_DIR, log_entry.TRAINING_HISTORY_Q_VALUE_LOG_ENTRY
            )
        )

        self.assertEqual(len(logged_entries), 18)
        for logged_entry, (
            expected_batch_idx,
            expected_state_id,
            expected_action,
        ) in zip(logged_entries, itertools.product([0, 1], [0, 2, 1], [0, 1, 2])):
            self.assertEqual(logged_entry.batch_idx, expected_batch_idx)
            self.assertEqual(logged_entry.state_id, expected_state_id)
            self.assertEqual(logged_entry.action, expected_action)

    def test_training_history_visit_logging(self):
        """Verifies that training history visits are logged in debug mode."""

        max_steps = 8
        with object_logging.ObjectLogManager(
            dirname=_OBJECT_LOGGING_DIR
        ) as object_log_manager:
            _get_trainer(max_steps=max_steps).fit(
                _get_model(
                    object_log_manager=object_log_manager,
                    debug=True,
                    max_episode_length=2,
                    initial_memories_count=1,
                )
            )

        expected_entries = [
            log_entry.OccurrenceLogEntry(batch_idx=batch_idx, object=state_id)
            for batch_idx, state_id in zip(
                range(-1, max_steps), [0, 0, 0, 2, 0, 3, 0, 3, 0]
            )
        ]

        logged_entries = list(
            object_log_readers.read_object_log(
                _OBJECT_LOGGING_DIR, log_entry.TRAINING_HISTORY_VISIT_LOG_ENTRY
            )
        )

        self._verify_log_entries(expected_entries, logged_entries)


class BuilderTest(unittest.TestCase):
    """Verifies the builder's execution of a policy."""

    def test_builder(self):
        """Checks an unsuccessful and successful build on the same env."""
        test_builder = builder.Builder(
            builder_trainer.make_env(
                name=_ENV_NAME, width=4, force_standard_config=True
            )
        )

        def _constant_estimator(state) -> torch.Tensor:
            """Returns a policy that always adds a brick to the left side."""
            return torch.tensor([1, 0, 0, 0])

        episode_length = 4
        build_result = test_builder.build(
            policy=policies.GreedyPolicy(_constant_estimator),
            episode_length=episode_length,
            render=False,
        )
        self.assertFalse(build_result.success)
        self.assertEqual(build_result.reward, -1 * episode_length)
        self.assertEqual(build_result.steps, episode_length)

        alternator = False

        def _alternating_estimator(state) -> torch.Tensor:
            """Returns a policy that alternates between adding a brick to the left side, then right side."""
            nonlocal alternator
            alternator = not alternator
            if alternator:
                return torch.tensor([1, 0, 0, 0])

            return torch.tensor([0, 0, 1, 0])

        episode_length = 4
        build_result = test_builder.build(
            policy=policies.GreedyPolicy(_alternating_estimator),
            episode_length=episode_length,
            render=False,
        )
        self.assertTrue(build_result.success)
        # The first gives -1 reward. Then we get a 100 completion bonus.
        self.assertEqual(build_result.reward, 99)
        self.assertEqual(build_result.steps, 2)


class BuildEvaluatorTest(unittest.TestCase):
    """Verifies the build evaluation metric computation."""

    def test_build_evaluator(self):
        """Checks metrics after some simple builds."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=4, force_standard_config=False, seed=12345
        )

        build_count = 5
        episode_length = 4

        alternator = False

        def _alternating_estimator(state) -> torch.Tensor:
            nonlocal alternator
            alternator = not alternator
            if alternator:
                return torch.tensor([1, 0, 0, 0])

            return torch.tensor([0, 0, 1, 0])

        build_evaluator = builder.BuildEvaluator(
            env=env,
            policy=policies.GreedyPolicy(_alternating_estimator),
            build_count=build_count,
            episode_length=episode_length,
        )

        # Stats manually verified from the following:
        #
        # [BuildResult(success=True, reward=99, steps=2, final_state=array([
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [2., 2., 2., 2.],
        # [1., 0., 0., 1.],
        # [1., 0., 0., 1.],
        # [1., 0., 0., 1.]])),
        # BuildResult(success=True, reward=97, steps=4, final_state=array([
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [2., 2., 0., 0.],
        # [2., 2., 2., 2.],
        # [1., 0., 2., 2.],
        # [1., 0., 1., 1.]])),
        # BuildResult(success=True, reward=99, steps=2, final_state=array([
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [2., 2., 0., 0.],
        # [1., 1., 2., 2.],
        # [1., 1., 0., 1.]])),
        # BuildResult(success=True, reward=100, steps=1, final_state=array([
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [0., 0., 0., 0.],
        # [2., 2., 1., 1.],
        # [1., 0., 1., 1.],
        # [1., 0., 1., 1.]])),
        # BuildResult(success=False, reward=-4, steps=4, final_state=array([
        # [0., 0., 0., 0.],
        # [2., 2., 0., 0.],
        # [2., 2., 0., 0.],
        # [1., 0., 2., 2.],
        # [1., 0., 2., 2.],
        # [1., 0., 0., 1.]]))]

        self.assertEqual(build_evaluator.success_rate, 0.8)
        np.testing.assert_array_equal(
            build_evaluator.successes, [True, True, True, True, False]
        )
        self.assertEqual(build_evaluator.build_steps_on_success_mean, 2.25)
        np.testing.assert_array_equal(build_evaluator.build_steps, [2, 4, 2, 1, 4])
        self.assertEqual(build_evaluator.reward_on_success_mean, 98.75)
        np.testing.assert_array_equal(build_evaluator.rewards, [99, 97, 99, 100, -4])
        self.assertEqual(build_evaluator.height_of_highest_block_mean, 2.8)


if __name__ == "__main__":
    unittest.main()
