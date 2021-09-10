"""Tests for core building and training components."""
import unittest

import itertools
from typing import List
from parameterized import parameterized
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping

import torch

from bridger import builder
from bridger import builder_trainer
from bridger import policies

_ENV_NAME = "gym_bridges.envs:Bridges-v0"


class BridgeBuilderTrainerTest(unittest.TestCase):
    """Verifies training hooks and structure."""

    def test_validation_builder(self):
        """Ensures ValidationBuilder keeps building and returning results."""
        env = builder_trainer.make_env(
            name=_ENV_NAME, width=4, force_standard_config=True
        )

        def _constant_estimator(state) -> torch.Tensor:
            return torch.tensor([1, 0, 0, 0])

        # Get an arbitrary number of results. The ValidationBuilder
        # will keep producing results until we stop asking.
        for build_result in itertools.islice(builder_trainer.ValidationBuilder(
            env=env, policy=policies.GreedyPolicy(_constant_estimator), episode_length=1
        ), 10):
            self.assertFalse(build_result[0])
            self.assertEqual(build_result[1], -1)
            self.assertEqual(build_result[2], 1)

    @parameterized.expand(
        [
            ("No Early Stopping", [], 5),
            (
                "Early Stopping",
                [
                    EarlyStopping(
                        monitor="val_reward", patience=1, mode="max", strict=True
                    )
                ],
                3,
            ),
        ]
    )
    def test_early_stopping(
        self,
        name: str,
        early_stopping_callback: List[Callback],
        expected_calls_count: int,
    ):
        """Checks early stopping callback actually stops training."""

        class CountingCallback(Callback):
            """Callback that counts the number of training batches."""

            def __init__(self):
                self.count = 0

            def on_train_batch_end(
                self, trainer, model, outputs, batch, batch_idx, dataloader_idx
            ):
                self.count += 1

        def get_model() -> builder_trainer.BridgeBuilderTrainer:
            return builder_trainer.BridgeBuilderTrainer.instantiate(
                env_width=3,
                env_force_standard_config=True,
                max_episode_length=1,
                val_batch_size=1,
            )

        def get_trainer(callbacks: List[Callback]) -> Trainer:
            return Trainer(
                val_check_interval=1,
                # The validation batch size can be adjusted via a config, but
                # we only need a single batch.
                limit_val_batches=1,
                max_steps=5,
                callbacks=callbacks,
            )

        callbacks = [CountingCallback()] + early_stopping_callback
        get_trainer(callbacks).fit(get_model())
        self.assertEqual(callbacks[0].count, expected_calls_count)


class BuilderTest(unittest.TestCase):
    """Verifies the builder's execution of a policy."""

    def test_builder(self):
        """Checks an unsucessful and successful build on the same env."""
        test_builder = builder.Builder(
            builder_trainer.make_env(
                name=_ENV_NAME, width=4, force_standard_config=True
            )
        )

        def _constant_estimator(state) -> torch.Tensor:
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


if __name__ == "__main__":
    unittest.main()
